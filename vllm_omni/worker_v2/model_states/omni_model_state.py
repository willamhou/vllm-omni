"""OmniModelState — generic ModelState base for all Omni model stages.

Extends ``DefaultModelState`` with:

* Cross-stage intermediate buffer (``OmniIntermediateBuffer``)
* ``model_intermediate_buffer`` / ``runtime_additional_information`` injection
  into ``model_inputs`` via ``prepare_inputs()``
* ``OmniOutput`` → ``(text_hidden, multimodal_outputs)`` post-processing
* Plugin lifecycle dispatch (``OmniModelStatePlugin``)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.model_states.intermediate_buffer import (
    OmniIntermediateBuffer,
)
from vllm_omni.worker_v2.model_states.plugin import OmniModelStatePlugin

logger = init_logger(__name__)


class OmniModelState(DefaultModelState):
    """Generic Omni ``ModelState`` — works for **all** Omni model stages.

    Model-specific behaviour is injected via ``OmniModelStatePlugin``
    instances or subclasses; this class itself is model-agnostic.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        max_num_reqs = self.scheduler_config.max_num_seqs
        self.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs)
        self.has_preprocess: bool = getattr(model, "has_preprocess", False)
        self.has_postprocess: bool = getattr(model, "has_postprocess", False)
        self.have_multimodal_outputs: bool = getattr(model, "have_multimodal_outputs", False)
        self.plugins: list[OmniModelStatePlugin] = []

        if hasattr(model, "get_omni_plugins"):
            for plugin in model.get_omni_plugins():
                self.register_plugin(plugin)

    # ------------------------------------------------------------------
    # Plugin management
    # ------------------------------------------------------------------

    def register_plugin(self, plugin: OmniModelStatePlugin) -> None:
        self.plugins.append(plugin)

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self.intermediate_buffer.add_request(req_index, new_req_data)
        for plugin in self.plugins:
            plugin.on_add_request(req_index, new_req_data)

    def remove_request(self, req_index: int) -> None:
        self.intermediate_buffer.remove_request(req_index)
        for plugin in self.plugins:
            plugin.on_remove_request(req_index)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        base = super().prepare_inputs(input_batch, req_states)
        buffer_list = self.intermediate_buffer.gather(input_batch)
        base["model_intermediate_buffer"] = buffer_list
        base["runtime_additional_information"] = buffer_list
        for plugin in self.plugins:
            base.update(plugin.prepare_extra_inputs(input_batch, req_states))
        return base

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        base = super().prepare_dummy_inputs(num_reqs, num_tokens)
        dummy_buffer = [{} for _ in range(num_reqs)]
        base["model_intermediate_buffer"] = dummy_buffer
        base["runtime_additional_information"] = dummy_buffer
        return base

    # ------------------------------------------------------------------
    # Output post-processing
    # ------------------------------------------------------------------

    def postprocess_model_output(
        self,
        model_output: Any,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> tuple[torch.Tensor, dict]:
        """Convert raw model output to ``(text_hidden, multimodal_outputs)``.

        Handles ``OmniOutput`` unwrapping and ``make_omni_output``
        conversion, then dispatches to registered plugins.
        """
        if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
            buffer_list = self.intermediate_buffer.gather(input_batch)
            model_output = self.model.make_omni_output(
                model_output,
                model_intermediate_buffer=buffer_list,
                runtime_additional_information=buffer_list,
            )

        if self.have_multimodal_outputs and isinstance(model_output, OmniOutput):
            text_hidden = model_output.text_hidden_states
            multimodal_outputs: dict = model_output.multimodal_outputs or {}
        elif isinstance(model_output, (list, tuple)):
            text_hidden = model_output[0]
            multimodal_outputs = {}
        else:
            text_hidden = model_output
            multimodal_outputs = {}

        for plugin in self.plugins:
            text_hidden, multimodal_outputs = plugin.postprocess(
                text_hidden, multimodal_outputs, input_batch, req_states
            )

        return text_hidden, multimodal_outputs
