"""OmniIntermediateBuffer — per-request cross-stage state for Omni pipelines.

Uses ``req_index`` (not ``req_id``) for O(1) access, aligned with v2's
``RequestState`` slot management.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


def _resolve_prompt_embeds(pe: Any) -> torch.Tensor | None:
    """Convert a prompt_embeds payload to a contiguous CPU tensor."""
    if pe is None:
        return None
    try:
        if isinstance(pe, torch.Tensor):
            return pe.detach().cpu().contiguous()
        data = getattr(pe, "data", None)
        shape = getattr(pe, "shape", None)
        if data is not None and shape is not None:
            dt = np.dtype(getattr(pe, "dtype", "float32"))
            arr = np.frombuffer(data, dtype=dt).reshape(shape)
            return torch.from_numpy(arr.copy())
    except Exception:
        logger.exception("Failed to decode prompt_embeds payload")
    return None


def _resolve_additional_information(payload: Any) -> dict[str, Any]:
    """Convert an additional_information payload to a plain dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    try:
        entries = getattr(payload, "entries", None)
        if not isinstance(entries, dict):
            return {}
        info: dict[str, Any] = {}
        for k, entry in entries.items():
            tensor_data = getattr(entry, "tensor_data", None)
            if tensor_data is not None:
                dt = np.dtype(getattr(entry, "tensor_dtype", "float32"))
                arr = np.frombuffer(tensor_data, dtype=dt)
                arr = arr.reshape(getattr(entry, "tensor_shape", ()))
                info[k] = torch.from_numpy(arr.copy())
            else:
                info[k] = getattr(entry, "list_data", None)
        return info
    except Exception:
        logger.exception("Failed to decode additional_information payload")
    return {}


class OmniIntermediateBuffer:
    """Per-request intermediate state for multi-stage Omni pipelines.

    Stores ``prompt_embeds``, ``additional_information``, ``mm_features``,
    ``req_id`` and any runtime updates written back by model postprocess.
    """

    def __init__(self, max_num_reqs: int):
        self.buffers: list[dict[str, Any]] = [{} for _ in range(max_num_reqs)]

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        info: dict[str, Any] = {}

        pe = getattr(new_req_data, "prompt_embeds", None)
        if pe is not None:
            pe_cpu = _resolve_prompt_embeds(pe)
            if pe_cpu is not None:
                info["prompt_embeds_cpu"] = pe_cpu

        ai = getattr(new_req_data, "additional_information", None)
        if ai is not None:
            info.update(_resolve_additional_information(ai))

        if new_req_data.mm_features:
            info["mm_features"] = new_req_data.mm_features

        info["req_id"] = new_req_data.req_id
        self.buffers[req_index] = info

    def remove_request(self, req_index: int) -> None:
        self.buffers[req_index] = {}

    def gather(self, input_batch: InputBatch) -> list[dict[str, Any]]:
        """Return buffer dicts in current batch order (via ``idx_mapping_np``)."""
        return [self.buffers[idx] for idx in input_batch.idx_mapping_np]

    def update(
        self,
        req_index: int,
        updates: dict[str, Any],
        gpu_resident_keys: set[str] | None = None,
    ) -> None:
        """Merge *updates* into the buffer at *req_index*.

        Tensors are detached; those whose key is **not** in
        *gpu_resident_keys* are moved to CPU.
        """
        if not updates:
            return
        gpu_keys = gpu_resident_keys or set()
        existing = self.buffers[req_index]
        for k, v in updates.items():
            if isinstance(v, torch.Tensor):
                if k in gpu_keys:
                    existing[k] = v.detach().clone()
                else:
                    existing[k] = v.detach().cpu().contiguous()
            elif isinstance(v, list):
                existing[k] = [
                    (item.detach().cpu().contiguous() if isinstance(item, torch.Tensor) else item) for item in v
                ]
            else:
                existing[k] = v
