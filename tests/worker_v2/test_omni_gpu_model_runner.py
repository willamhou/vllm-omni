"""Unit tests for OmniGPUModelRunner v2 overrides."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyInputBatch:
    def __init__(self, idx_mapping=None):
        self.idx_mapping_np = idx_mapping or []


def _make_runner():
    """Create an OmniGPUModelRunner without calling __init__."""
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = MagicMock()
    runner.req_states = SimpleNamespace(req_id_to_index={"r1": 0, "r2": 1})
    runner.execute_model_state = None
    return runner


def test_finish_requests_calls_remove_for_finished():
    runner = _make_runner()
    mock_state = MagicMock()
    runner.model_state = mock_state

    sched_output = SimpleNamespace(
        finished_req_ids={"r1"},
        preempted_req_ids=set(),
    )

    with patch.object(type(runner).__bases__[0], "finish_requests", return_value=None):
        runner.finish_requests(sched_output)

    mock_state.remove_request.assert_called_once_with(0)


def test_finish_requests_calls_remove_for_preempted():
    runner = _make_runner()
    mock_state = MagicMock()
    runner.model_state = mock_state

    sched_output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"r2"},
    )

    with patch.object(type(runner).__bases__[0], "finish_requests", return_value=None):
        runner.finish_requests(sched_output)

    mock_state.remove_request.assert_called_once_with(1)


def test_finish_requests_ignores_unknown_req_ids():
    runner = _make_runner()
    mock_state = MagicMock()
    runner.model_state = mock_state

    sched_output = SimpleNamespace(
        finished_req_ids={"unknown"},
        preempted_req_ids=set(),
    )

    with patch.object(type(runner).__bases__[0], "finish_requests", return_value=None):
        runner.finish_requests(sched_output)

    mock_state.remove_request.assert_not_called()


def test_finish_requests_handles_both_finished_and_preempted():
    runner = _make_runner()
    mock_state = MagicMock()
    runner.model_state = mock_state

    sched_output = SimpleNamespace(
        finished_req_ids={"r1"},
        preempted_req_ids={"r2"},
    )

    with patch.object(type(runner).__bases__[0], "finish_requests", return_value=None):
        runner.finish_requests(sched_output)

    assert mock_state.remove_request.call_count == 2
