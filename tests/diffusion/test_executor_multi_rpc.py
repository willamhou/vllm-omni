# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for sender-driven demux in MultiprocDiffusionExecutor (RFC #3158 Step A).

Verifies that concurrent collective_rpc callers correctly route their
replies through the shared broadcast/result MQ pair using a monotonic
``correlation_id`` envelope, without swapping replies between callers
or dead-locking under contention.

These tests do NOT spawn real workers; they mirror the lightweight
``_make_executor`` pattern from ``test_multiproc_engine_concurrency.py``.

Stubs in this file are skipped until the production code lands. The
sequencing is:

* Step 2 unblocks tests 1, 7 (envelope + correlation IDs).
* Step 3 unblocks tests 2, 3, 6, 8 (sender-driven demux).
* Step 4 unblocks tests 5, 9, 10 (cleanup paths + serialize lock).
* Step 5 unblocks tests 4, 11 (serialize-lock + benchmark smoke).
"""

from __future__ import annotations

import itertools
import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


# ───────────────────────────── helpers ─────────────────────────────


def _tagged_output(tag: str) -> DiffusionOutput:
    """Return a ``DiffusionOutput`` identifiable by its ``error`` field."""
    return DiffusionOutput(output=torch.tensor([0]), error=tag)


def _make_executor(num_gpus: int = 1):
    """Build a ``MultiprocDiffusionExecutor`` without spawning workers.

    Returns ``(executor, request_queue, result_queue)``. The dispatcher
    state attributes (``_cid_counter``, ``_pending``, ``_inbox``, ...)
    are set defensively here so the fixture remains usable while the
    production code is still being written across Steps 2–4.
    """
    od_cfg = SimpleNamespace(num_gpus=num_gpus)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(MultiprocDiffusionExecutor, "_init_executor", lambda self: None)
    executor = MultiprocDiffusionExecutor(od_cfg)
    monkeypatch.undo()

    req_q: queue.Queue = queue.Queue()
    res_q: queue.Queue = queue.Queue()

    executor._broadcast_mq = SimpleNamespace(enqueue=req_q.put)
    executor._result_mq = SimpleNamespace(
        dequeue=lambda timeout=None: res_q.get(timeout=timeout if timeout is not None else 10),
    )
    executor._closed = False
    executor._processes = []
    executor.is_failed = False
    executor._failure_callbacks = []
    # Mirror state that ``_init_executor`` would normally populate.
    executor._cid_counter = itertools.count(1)
    executor._cid_lock = threading.Lock()
    return executor, req_q, res_q


# ───────────────────────── Step 2: envelope + correlation IDs ─────────────────


class TestCorrelationIDs:
    def test_correlation_id_is_monotonic_and_unique(self) -> None:
        """1k sequential RPCs produce strictly increasing correlation IDs
        with no duplicates and no gaps under normal flow."""
        executor, req_q, res_q = _make_executor()

        n = 1000
        cids: list[int] = []

        def fake_worker():
            for _ in range(n):
                req = req_q.get(timeout=10)
                cid = req["correlation_id"]
                cids.append(cid)
                # Echo back as proper rpc_reply envelope
                res_q.put({"type": "rpc_reply", "correlation_id": cid, "payload": _tagged_output(f"r{cid}")})

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        for _ in range(n):
            executor.collective_rpc("ping", unique_reply_rank=0)

        worker.join(timeout=10)
        assert cids == sorted(cids), "correlation_ids must be monotonically increasing"
        assert len(set(cids)) == n, "correlation_ids must be unique"
        assert cids[-1] - cids[0] == n - 1, "no gaps expected for sequential calls"

    def test_backward_compat_reply_without_correlation_id_routed_to_oldest(self) -> None:
        """A bare reply (no envelope) must still satisfy the in-flight
        single-flight call. Codex Q2 verified this is correct because
        WorkerProc.return_result is single-threaded and rank-0 only, so
        legacy untagged replies remain single-source FIFO."""
        executor, req_q, res_q = _make_executor()

        def fake_worker():
            req_q.get(timeout=10)
            # Reply WITHOUT envelope (legacy worker path / add_req path)
            res_q.put(_tagged_output("legacy"))

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        result = executor.collective_rpc("ping", unique_reply_rank=0)
        worker.join(timeout=5)

        assert isinstance(result, DiffusionOutput)
        assert result.error == "legacy", "legacy untagged reply must be returned to caller"


# ───────────────────────── Step 3: sender-driven demux ────────────────────────


class TestSenderDrivenDemux:
    def test_concurrent_collective_rpc_results_routed_correctly(self) -> None:
        """32 concurrent collective_rpc callers each receive their own
        reply (no swap) — the sender-driven demuxer correctly stashes
        off-target replies into ``_inbox`` for the true owner."""
        pytest.skip("Step 3: sender-driven demux not yet implemented")

    def test_concurrent_request_and_collective_rpc_no_swap(self) -> None:
        """Mix execute_request and collective_rpc('ping') 16-deep
        concurrently; both sides return correct payloads."""
        pytest.skip("Step 3: sender-driven demux not yet implemented")

    def test_late_reply_for_dead_cid_dropped(self) -> None:
        """A reply with a correlation_id not in ``_pending`` is logged
        with WARNING and dropped silently — does not crash the
        dequeuer."""
        pytest.skip("Step 3: dispatch-cond + inbox handling not yet implemented")

    def test_dequeue_role_no_starvation(self) -> None:
        """Across 100 sequential RPCs from a rotating set of caller
        threads, no thread waits more than 2× the median for the
        ``_dequeue_lock``. Dequeuer role rotates fairly."""
        pytest.skip("Step 3: dequeue lock + condition wait not yet implemented")


# ───────────────────────── Step 4: cleanup paths + serialize lock ────────────


class TestCleanupPaths:
    def test_timeout_drains_pending_set(self) -> None:
        """A caller that hits TimeoutError must remove its cid from
        ``_pending`` (via the ``finally`` block) so subsequent calls do
        not see leaked entries."""
        pytest.skip("Step 4: finally-block cleanup not yet implemented")

    def test_inbox_drained_on_caller_timeout(self) -> None:
        """If a reply lands in ``_inbox`` for a cid whose caller has
        already timed out, the ``finally`` block GCs the entry so the
        inbox does not grow unboundedly."""
        pytest.skip("Step 4: inbox GC in finally not yet implemented")

    def test_engine_dead_propagates_to_dequeuer_and_waiters(self) -> None:
        """When ``is_failed`` flips True mid-flight, the active dequeuer
        raises ``EngineDeadError`` and ``_dispatch_cond.notify_all()`` in
        ``shutdown`` wakes any condition-waiting peers so they re-check
        ``is_failed`` and raise too."""
        pytest.skip("Step 4: engine-dead fan-out not yet implemented")


# ─────────────────── Step 5: serialize lock + benchmark smoke ────────────────


class TestSerializeAndBench:
    def test_serialized_methods_block_concurrency(self) -> None:
        """Methods in ``_NON_INTERLEAVABLE_METHODS`` (sleep / wake_up /
        handle_sleep_task / handle_wake_task) acquire ``_serialize_lock``
        at the head of ``collective_rpc`` so they cannot interleave with
        each other or with other in-flight RPCs that hold it."""
        pytest.skip("Step 5: serialize lock not yet implemented")

    def test_benchmark_smoke_overhead_under_budget(self) -> None:
        """200 sequential no-op RPCs through the demux loop add a
        median latency of <50µs vs a baseline that bypasses cid
        stamping. Smoke-grade; the real benchmark is the standalone
        script ``benchmarks/diffusion/bench_executor_demux.py``."""
        pytest.skip("Step 5: benchmark smoke not yet implemented")
