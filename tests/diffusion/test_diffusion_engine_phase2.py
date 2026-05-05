# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DiffusionEngine Phase 2 fixes: CPU offload, audio slicing, metadata."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import torch


def _load_engine_module():
    """Load diffusion_engine.py with mocked heavy dependencies."""
    engine_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "vllm_omni",
            "diffusion",
            "diffusion_engine.py",
        )
    )

    mocks = {
        "vllm_omni": MagicMock(),
        "vllm_omni.diffusion": types.ModuleType("vllm_omni.diffusion"),
        "vllm_omni.diffusion.data": MagicMock(),
        "vllm_omni.diffusion.executor": MagicMock(),
        "vllm_omni.diffusion.executor.abstract": MagicMock(),
        "vllm_omni.diffusion.registry": MagicMock(),
        "vllm_omni.diffusion.request": MagicMock(),
        "vllm_omni.diffusion.sched": MagicMock(),
        "vllm_omni.diffusion.sched.interface": MagicMock(),
        "vllm_omni.diffusion.worker": MagicMock(),
        "vllm_omni.diffusion.worker.utils": MagicMock(),
        "vllm_omni.inputs": MagicMock(),
        "vllm_omni.inputs.data": MagicMock(),
        "vllm_omni.outputs": MagicMock(),
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.exceptions": MagicMock(),
        "vllm.logger": MagicMock(init_logger=lambda name: MagicMock()),
        "PIL": MagicMock(),
        "PIL.Image": MagicMock(),
    }
    with patch.dict(sys.modules, mocks):
        spec = importlib.util.spec_from_file_location("diffusion_engine", engine_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


_engine_mod = _load_engine_module()
_move_tensors_to_cpu = _engine_mod._move_tensors_to_cpu
_normalize_outputs = _engine_mod._normalize_outputs


class TestMoveTensorsToCpu:
    """Test recursive CPU offload helper."""

    def test_single_tensor(self) -> None:
        t = torch.randn(2, 3)
        result = _move_tensors_to_cpu(t)
        assert result.device.type == "cpu"
        assert torch.equal(result, t)

    def test_already_on_cpu(self) -> None:
        t = torch.randn(2, 3)
        result = _move_tensors_to_cpu(t)
        # Should return as-is (no copy needed)
        assert result is t

    def test_dict_with_tensors(self) -> None:
        data = {"video": torch.randn(2, 3), "audio": torch.randn(4), "label": "test"}
        result = _move_tensors_to_cpu(data)
        assert isinstance(result, dict)
        assert result["video"].device.type == "cpu"
        assert result["audio"].device.type == "cpu"
        assert result["label"] == "test"

    def test_tuple_with_tensors(self) -> None:
        data = (torch.randn(2, 3), torch.randn(4))
        result = _move_tensors_to_cpu(data)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].device.type == "cpu"
        assert result[1].device.type == "cpu"

    def test_list_with_tensors(self) -> None:
        data = [torch.randn(2, 3), torch.randn(4)]
        result = _move_tensors_to_cpu(data)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_nested_structure(self) -> None:
        data = {"outputs": (torch.randn(2, 3), {"inner": torch.randn(4)})}
        result = _move_tensors_to_cpu(data)
        assert result["outputs"][1]["inner"].device.type == "cpu"

    def test_non_tensor_passthrough(self) -> None:
        assert _move_tensors_to_cpu(42) == 42
        assert _move_tensors_to_cpu("hello") == "hello"
        assert _move_tensors_to_cpu(None) is None


class TestBatchedOutputSlicing:
    """Test that batched tensor outputs are properly split per-prompt."""

    def test_batched_tensor_is_split(self) -> None:
        """A batched tensor should be split when item count matches."""
        outputs = torch.randn(3, 16000)  # 3 audio samples batched
        outputs = _normalize_outputs(outputs, expected_items=3)
        assert len(outputs) == 3
        assert outputs[0].shape == (16000,)

    def test_single_sample_not_split(self) -> None:
        """A single sample tensor should remain one item."""
        outputs = torch.randn(1, 16000)
        outputs = _normalize_outputs(outputs, expected_items=1)
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 16000)

    def test_numpy_batched_is_split(self) -> None:
        """Batched numpy arrays should also be split."""
        outputs = np.random.randn(3, 16000)
        outputs = _normalize_outputs(outputs, expected_items=3)
        assert len(outputs) == 3

    def test_list_passthrough(self) -> None:
        """A list output should not be modified."""
        outputs = [torch.randn(16000), torch.randn(16000)]
        normalized = _normalize_outputs(outputs, expected_items=2)
        assert normalized is outputs

    def test_shape_mismatch_is_not_split(self) -> None:
        """A 1D waveform should not be mistaken for a batch."""
        outputs = torch.randn(16000)
        normalized = _normalize_outputs(outputs, expected_items=2)
        assert len(normalized) == 1
        assert normalized[0].shape == (16000,)
