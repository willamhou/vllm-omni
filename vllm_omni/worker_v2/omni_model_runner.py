"""OmniGPUModelRunner — thin inheritance layer over v2 GPUModelRunner.

Injects ``OmniModelState`` via ``load_model`` and adds a
``finish_requests`` hook to clean up the intermediate buffer.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

logger = init_logger(__name__)


class OmniGPUModelRunner(GPUModelRunner):
    """Thin layer over v2 ``GPUModelRunner`` for Omni lifecycle hooks."""

    model_state: OmniModelState

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)
        self.model_state = init_omni_model_state(self.vllm_config, self.model, self.encoder_cache, self.device)

    # ------------------------------------------------------------------
    # Request lifecycle: clean up intermediate buffer on finish
    # ------------------------------------------------------------------

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished = scheduler_output.finished_req_ids
        preempted = scheduler_output.preempted_req_ids
        all_done = finished | preempted if preempted else finished
        for req_id in all_done:
            idx = self.req_states.req_id_to_index.get(req_id)
            if idx is not None:
                self.model_state.remove_request(idx)
        super().finish_requests(scheduler_output)

    # ------------------------------------------------------------------
    # Output intercept: wrap non-Tensor hidden_states → OmniOutput
    # ------------------------------------------------------------------

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Any | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> Any:
        result = super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
        )
        if self.execute_model_state is not None:
            hs = self.execute_model_state.hidden_states
            if not isinstance(hs, torch.Tensor) and hasattr(self.model, "make_omni_output"):
                buffer_list = self.model_state.intermediate_buffer.gather(self.execute_model_state.input_batch)
                omni_output = self.model.make_omni_output(
                    hs,
                    model_intermediate_buffer=buffer_list,
                    runtime_additional_information=buffer_list,
                )
                self.execute_model_state = self.execute_model_state._replace(hidden_states=omni_output)
        return result
