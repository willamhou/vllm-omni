"""Unit tests for OmniARModelRunner v2."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_pooler_output_basic():
    """Verify _build_pooler_output slices per-request hidden + mm."""
    runner = object.__new__(OmniARModelRunner)

    hidden = torch.randn(6, 8)
    mm = {"audio": torch.randn(6, 2)}

    batch = SimpleNamespace(
        req_ids=["r1", "r2"],
        query_start_loc_np=[0, 3],
        num_scheduled_tokens=[3, 3],
    )

    pooler = OmniARModelRunner._build_pooler_output(runner, hidden, mm, batch)

    assert len(pooler) == 2
    assert pooler[0]["hidden"].shape == (3, 8)
    assert pooler[1]["hidden"].shape == (3, 8)
    assert pooler[0]["audio"].shape == (3, 2)


def test_build_pooler_output_empty_mm():
    runner = object.__new__(OmniARModelRunner)

    hidden = torch.randn(4, 8)
    batch = SimpleNamespace(
        req_ids=["r1"],
        query_start_loc_np=[0],
        num_scheduled_tokens=[4],
    )

    pooler = OmniARModelRunner._build_pooler_output(runner, hidden, {}, batch)
    assert len(pooler) == 1
    assert "hidden" in pooler[0]
    assert len(pooler[0]) == 1


def test_copy_mm_to_cpu_tensor():
    total = 10
    t = torch.randn(10, 4)
    result = OmniARModelRunner._copy_mm_to_cpu({"feat": t}, total)
    assert "feat" in result
    assert result["feat"].shape == (10, 4)
    assert result["feat"].device == torch.device("cpu")


def test_copy_mm_to_cpu_dict():
    total = 10
    d = {"inner": torch.randn(10, 2)}
    result = OmniARModelRunner._copy_mm_to_cpu({"nested": d}, total)
    assert "nested" in result
    assert "inner" in result["nested"]


def test_copy_mm_to_cpu_list():
    result = OmniARModelRunner._copy_mm_to_cpu({"items": [torch.randn(3), "text"]}, 10)
    assert "items" in result
    assert isinstance(result["items"][0], torch.Tensor)
    assert result["items"][1] == "text"


def test_copy_mm_to_cpu_empty():
    assert OmniARModelRunner._copy_mm_to_cpu({}, 10) == {}


def test_slice_mm_payload_tensor():
    hidden = torch.randn(6, 4)
    mm_cpu = {"feat": torch.randn(6, 2)}
    result = OmniARModelRunner._slice_mm_payload(mm_cpu, 0, 3, 0, hidden)
    assert result["feat"].shape == (3, 2)


def test_slice_mm_payload_list():
    hidden = torch.randn(6, 4)
    mm_cpu = {"items": [torch.randn(2), torch.randn(3)]}
    result = OmniARModelRunner._slice_mm_payload(mm_cpu, 0, 3, 0, hidden)
    assert isinstance(result["items"], torch.Tensor)


def test_slice_mm_payload_dict():
    hidden = torch.randn(6, 4)
    mm_cpu = {"nested": {"a": torch.randn(6, 2)}}
    result = OmniARModelRunner._slice_mm_payload(mm_cpu, 2, 5, 1, hidden)
    assert result["nested"]["a"].shape == (3, 2)
