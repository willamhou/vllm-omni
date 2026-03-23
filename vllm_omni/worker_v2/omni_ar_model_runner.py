"""OmniARModelRunner — autoregressive stage runner on MR V2.

Extends ``OmniGPUModelRunner`` with:

* ``OmniOutput`` post-processing in ``sample_tokens``
* Per-request ``pooler_output`` construction (hidden + multimodal slices)
* Cross-stage KV extraction before state cleanup
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVTransferManager,
)
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

logger = init_logger(__name__)

_EMPTY = OmniModelRunnerOutput(req_ids=[], req_id_to_index={})


class OmniARModelRunner(OmniGPUModelRunner):
    """AR stage runner. Produces per-request hidden states + multimodal outputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
        self._kv_extracted_req_ids: list[str] | None = None

    # ------------------------------------------------------------------
    # execute_model: KV transfer pre-hook + delegate to super
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Any | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> Any:
        self._handle_kv_transfer_pre(scheduler_output)
        return super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
        )

    # ------------------------------------------------------------------
    # sample_tokens: OmniOutput handling + pooler_output
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput | None) -> OmniModelRunnerOutput | ModelRunnerOutput | None:
        kv_extracted = self._kv_extracted_req_ids
        self._kv_extracted_req_ids = None

        if self.execute_model_state is None:
            return None

        input_batch = self.execute_model_state.input_batch
        hidden_states = self.execute_model_state.hidden_states
        kv_connector_output = self.execute_model_state.kv_connector_output
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            from vllm.v1.worker.gpu.pp_utils import pp_receive

            sampled, num_sampled, num_rejected = pp_receive(
                input_batch.num_reqs,
                max_sample_len=self.num_speculative_steps + 1,
            )
            self.postprocess(input_batch, sampled, num_sampled, num_rejected)
            return None

        # --- Omni: post-process model output ---
        text_hidden, multimodal_outputs = self.model_state.postprocess_model_output(
            hidden_states, input_batch, self.req_states
        )

        # --- Standard v2 sampling ---
        sampler_output, num_sampled, num_rejected = self.sample(text_hidden, input_batch, grammar_output)

        if self.use_pp:
            from vllm.v1.worker.gpu.pp_utils import pp_broadcast

            pp_broadcast(sampler_output.sampled_token_ids, num_sampled, num_rejected)

        # --- Omni: prompt logprobs ---
        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            text_hidden,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
            self.req_states.prefill_len.np,
            self.req_states.num_computed_prefill_tokens,
        )

        # --- Omni: pooler_output ---
        engine_output_type = getattr(self.vllm_config.model_config, "engine_output_type", "text")
        if engine_output_type != "text":
            pooler_output = self._build_pooler_output(text_hidden, multimodal_outputs, input_batch)
        else:
            pooler_output = None

        # --- Postprocess (Triton kernel updates req_states) ---
        self.postprocess(
            input_batch,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
        )

        output = OmniModelRunnerOutput(
            req_ids=input_batch.req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(input_batch.req_ids)},
            sampled_token_ids=sampler_output.sampled_token_ids,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
        )
        output.kv_extracted_req_ids = kv_extracted
        return output

    # ------------------------------------------------------------------
    # pooler_output construction
    # ------------------------------------------------------------------

    def _build_pooler_output(
        self,
        text_hidden: torch.Tensor,
        multimodal_outputs: dict,
        input_batch: Any,
    ) -> list[dict[str, Any]]:
        """Slice per-request hidden states + multimodal outputs to CPU."""
        hidden_cpu = text_hidden.detach().cpu().contiguous()
        mm_cpu = self._copy_mm_to_cpu(multimodal_outputs, hidden_cpu.shape[0])

        query_start = input_batch.query_start_loc_np
        num_sched = input_batch.num_scheduled_tokens

        pooler: list[dict[str, Any]] = []
        for i, _req_id in enumerate(input_batch.req_ids):
            start = int(query_start[i])
            end = start + int(num_sched[i])
            payload: dict[str, Any] = {"hidden": hidden_cpu[start:end]}
            if mm_cpu:
                payload.update(self._slice_mm_payload(mm_cpu, start, end, i, hidden_cpu))
            pooler.append(payload)
        return pooler

    @staticmethod
    def _copy_mm_to_cpu(mm_outputs: dict, total_tokens: int) -> dict[str, Any]:
        cpu: dict[str, Any] = {}
        if not mm_outputs:
            return cpu
        for k, v in mm_outputs.items():
            try:
                if isinstance(v, torch.Tensor) and v.shape[0] == total_tokens:
                    cpu[k] = v.detach().cpu().contiguous()
                elif isinstance(v, dict):
                    sub: dict[str, torch.Tensor] = {}
                    for sk, sv in v.items():
                        if isinstance(sv, torch.Tensor) and sv.shape[0] == total_tokens:
                            sub[str(sk)] = sv.detach().cpu().contiguous()
                    if sub:
                        cpu[k] = sub
                elif isinstance(v, list) and v:
                    cpu[k] = [(el.detach().cpu().contiguous() if isinstance(el, torch.Tensor) else el) for el in v]
            except Exception:
                logger.exception("Error copying multimodal output %s to CPU", k)
        return cpu

    @staticmethod
    def _slice_mm_payload(
        mm_cpu: dict[str, Any],
        start: int,
        end: int,
        req_idx: int,
        hidden_cpu: torch.Tensor,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        total = hidden_cpu.shape[0]
        for k, v in mm_cpu.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == total:
                payload[k] = v[start:end].contiguous()
            elif isinstance(v, dict):
                payload[k] = {sk: sv[start:end].contiguous() for sk, sv in v.items()}
            elif isinstance(v, list):
                elem = v[req_idx] if req_idx < len(v) else v[0]
                if isinstance(elem, torch.Tensor):
                    elem = elem.clone()
                payload[k] = elem
            else:
                payload[k] = v
        return payload

    # ------------------------------------------------------------------
    # KV transfer
    # ------------------------------------------------------------------

    def _handle_kv_transfer_pre(self, scheduler_output: SchedulerOutput) -> None:
        finished: dict = getattr(scheduler_output, "finished_requests_needing_kv_transfer", {})
        if finished and hasattr(self.model, "get_kv_transfer_metadata"):
            for req_id, data in finished.items():
                try:
                    meta = self.model.get_kv_transfer_metadata(req_id)
                    if meta:
                        existing = data.get("custom_metadata") or {}
                        existing.update(meta)
                        data["custom_metadata"] = existing
                except Exception:
                    logger.warning(
                        "Failed to get KV transfer metadata for %s",
                        req_id,
                        exc_info=True,
                    )
        self._kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(
            finished_reqs=finished,
            kv_caches=self.kv_caches,
            block_size=self.cache_config.block_size,
            cache_dtype=str(self.cache_config.cache_dtype),
        )
