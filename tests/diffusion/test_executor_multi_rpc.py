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

    def _fake_dequeue(timeout=None):
        try:
            return res_q.get(timeout=timeout if timeout is not None else 10)
        except queue.Empty as e:
            # Production MessageQueue raises TimeoutError on empty wait;
            # mirror that here so callers in the executor see the same
            # exception shape regardless of the underlying fixture.
            raise TimeoutError() from e

    executor._result_mq = SimpleNamespace(dequeue=_fake_dequeue)
    executor._closed = False
    executor._processes = []
    executor.is_failed = False
    executor._failure_callbacks = []
    # ``_cid_counter`` / ``_dispatch_lock`` / ``_serialize_lock`` are
    # already initialised by ``__init__`` -> ``_ensure_dispatch_state``
    # which runs before ``_init_executor`` is monkeypatched away.
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
        off-target replies into ``_inbox`` for the true owner.

        The worker echoes the caller's ``args[0]`` in the reply tag so a
        pair-swap (cid 1 receives cid 2's reply) would surface as a
        wrong ``caller{i}`` prefix on caller ``i``.
        """
        executor, req_q, res_q = _make_executor()
        n = 32

        # Worker thread: read all broadcasts, then reply in REVERSED order
        # to force off-target deliveries that exercise the inbox path.
        def fake_worker():
            requests = []
            for _ in range(n):
                requests.append(req_q.get(timeout=10))
            for req in reversed(requests):
                cid = req["correlation_id"]
                arg = req["args"][0]
                tag = f"caller{arg}_r{cid}"
                res_q.put({"type": "rpc_reply", "correlation_id": cid, "payload": _tagged_output(tag)})

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        results: dict[int, str] = {}
        results_lock = threading.Lock()

        def caller(i: int) -> None:
            r = executor.collective_rpc("ping", args=(i,), unique_reply_rank=0)
            with results_lock:
                results[i] = r.error

        callers = [threading.Thread(target=caller, args=(i,)) for i in range(n)]
        for t in callers:
            t.start()
        for t in callers:
            t.join(timeout=15)
        worker.join(timeout=5)

        assert len(results) == n, f"got {len(results)} results, expected {n}"
        # Per-caller identity check catches both bulk and pair swaps:
        # every caller must see a tag prefixed with its own caller id.
        for caller_id, tag in results.items():
            assert tag.startswith(f"caller{caller_id}_"), f"caller {caller_id} got {tag!r} — reply swap detected"
        # Tag-count must equal caller count (defence in depth).
        assert len(set(results.values())) == n, "duplicate tags imply reply swap"

    def test_concurrent_request_and_collective_rpc_no_swap(self) -> None:
        """Mix collective_rpc('execute_model') and collective_rpc('ping')
        16-deep concurrently; both sides return correct payloads
        addressed to the right caller."""
        executor, req_q, res_q = _make_executor()
        n_per_kind = 8

        def fake_worker():
            for _ in range(n_per_kind * 2):
                req = req_q.get(timeout=10)
                cid = req["correlation_id"]
                tag = f"{req['method']}_r{cid}"
                res_q.put({"type": "rpc_reply", "correlation_id": cid, "payload": _tagged_output(tag)})

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        results: list[tuple[str, str]] = []
        rlock = threading.Lock()

        def caller(method: str, idx: int) -> None:
            r = executor.collective_rpc(method, args=(idx,), unique_reply_rank=0)
            with rlock:
                results.append((method, r.error))

        threads = []
        for i in range(n_per_kind):
            threads.append(threading.Thread(target=caller, args=("execute_model", i)))
            threads.append(threading.Thread(target=caller, args=("ping", i)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        worker.join(timeout=5)

        # Every result tag must start with the method that produced it.
        for method, tag in results:
            assert tag.startswith(method + "_"), f"swap: {method} caller got {tag!r}"
        assert len(results) == n_per_kind * 2

    def test_late_reply_for_dead_cid_dropped(self) -> None:
        """A reply with a correlation_id not in ``_pending`` is logged
        with WARNING and dropped silently — does not crash the
        dequeuer."""
        executor, req_q, res_q = _make_executor()

        def fake_worker():
            req = req_q.get(timeout=10)
            cid = req["correlation_id"]
            # First push a dead-cid reply; dispatcher should drop it.
            res_q.put({"type": "rpc_reply", "correlation_id": 99999, "payload": _tagged_output("ghost")})
            # Then the real reply.
            res_q.put({"type": "rpc_reply", "correlation_id": cid, "payload": _tagged_output("real")})

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        result = executor.collective_rpc("ping", unique_reply_rank=0)
        worker.join(timeout=5)

        assert result.error == "real", "ghost reply must be dropped, real one returned"

    def test_dequeue_role_no_starvation(self) -> None:
        """Across many concurrent RPCs, every caller eventually gets a
        reply within a generous deadline — i.e. no caller is permanently
        starved by the dequeue-lock acquisition pattern."""
        executor, req_q, res_q = _make_executor()
        n = 16

        def fake_worker():
            for _ in range(n):
                req = req_q.get(timeout=10)
                cid = req["correlation_id"]
                res_q.put({"type": "rpc_reply", "correlation_id": cid, "payload": _tagged_output(f"r{cid}")})

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        finished = [False] * n

        def caller(i: int) -> None:
            r = executor.collective_rpc("ping", unique_reply_rank=0, timeout=10.0)
            assert r.error.startswith("r")
            finished[i] = True

        threads = [threading.Thread(target=caller, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
        worker.join(timeout=5)

        assert all(finished), f"starved callers: {[i for i, ok in enumerate(finished) if not ok]}"


# ───────────────────────── Step 4: cleanup paths + serialize lock ────────────


class TestCleanupPaths:
    def test_timeout_drains_pending_set(self) -> None:
        """A caller that hits TimeoutError must remove its cid from
        ``_pending`` (via the ``finally`` block) so subsequent calls do
        not see leaked entries."""
        executor, req_q, _ = _make_executor()

        # Worker drains the request but never replies → caller times out.
        def fake_worker():
            req_q.get(timeout=10)

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        with pytest.raises(TimeoutError):
            executor.collective_rpc("ping", unique_reply_rank=0, timeout=0.2)
        worker.join(timeout=5)

        assert executor._pending == set(), f"_pending leaked: {executor._pending}"
        assert executor._inbox == {}, f"_inbox leaked: {executor._inbox}"

    def test_inbox_drained_on_caller_timeout(self) -> None:
        """Across both successful and timed-out paths, ``_inbox`` and
        ``_pending`` are fully drained — no per-caller state leaks past
        the ``finally`` block."""
        executor, req_q, res_q = _make_executor()

        def fake_worker():
            # Respond to the first request, ignore the second.
            req = req_q.get(timeout=10)
            cid = req["correlation_id"]
            res_q.put(
                {
                    "type": "rpc_reply",
                    "correlation_id": cid,
                    "payload": _tagged_output("ok"),
                }
            )
            req_q.get(timeout=10)  # consume second, never reply

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        # Happy path
        r = executor.collective_rpc("ping", unique_reply_rank=0, timeout=1.0)
        assert r.error == "ok"

        # Timeout path
        with pytest.raises(TimeoutError):
            executor.collective_rpc("ping", unique_reply_rank=0, timeout=0.2)

        worker.join(timeout=5)
        assert executor._pending == set(), f"_pending leaked: {executor._pending}"
        assert executor._inbox == {}, f"_inbox leaked: {executor._inbox}"

    def test_engine_dead_propagates_to_dequeuer_and_waiters(self) -> None:
        """When ``is_failed`` is True, ``collective_rpc`` raises
        ``EngineDeadError`` for every concurrent caller. The dequeuer
        observes worker death directly via ``_dequeue_one_chunk``;
        peers parked on ``_dispatch_cond`` re-check ``is_failed`` after
        the dequeuer's ``finally`` ``notify_all`` wakes them."""
        from vllm.v1.engine.exceptions import EngineDeadError

        executor, _, _ = _make_executor()

        def _instant_timeout(timeout=None):
            raise TimeoutError()

        executor._result_mq.dequeue = _instant_timeout
        executor.is_failed = True

        outs: list[str] = []
        outs_lock = threading.Lock()

        def caller() -> None:
            try:
                executor.collective_rpc("ping", unique_reply_rank=0, timeout=1.0)
                with outs_lock:
                    outs.append("ok")
            except EngineDeadError:
                with outs_lock:
                    outs.append("dead")
            except BaseException as exc:  # noqa: BLE001
                with outs_lock:
                    outs.append(type(exc).__name__)

        threads = [threading.Thread(target=caller, daemon=True) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert outs.count("dead") == 3, f"all callers must raise EngineDeadError; got {outs}"


# ─────────────────── Step 5: serialize lock + benchmark smoke ────────────────


class TestSerializeAndBench:
    def test_serialized_methods_block_concurrency(self) -> None:
        """Methods in ``_NON_INTERLEAVABLE_METHODS`` (sleep / wake_up /
        handle_sleep_task / handle_wake_task) acquire ``_serialize_lock``
        at the head of ``collective_rpc`` so two such calls cannot run
        concurrently."""
        import time as _time

        executor, req_q, res_q = _make_executor()

        enqueue_times: list[float] = []
        orig_enqueue = executor._broadcast_mq.enqueue

        def _timed_enqueue(req):
            enqueue_times.append(_time.monotonic())
            orig_enqueue(req)

        executor._broadcast_mq.enqueue = _timed_enqueue

        def fake_worker():
            for _ in range(2):
                req = req_q.get(timeout=10)
                cid = req["correlation_id"]
                # Simulate worker doing 100ms of work; serialised
                # callers cannot start their broadcast until the
                # previous one's reply has been consumed.
                _time.sleep(0.1)
                res_q.put(
                    {
                        "type": "rpc_reply",
                        "correlation_id": cid,
                        "payload": _tagged_output(req["method"]),
                    }
                )

        worker = threading.Thread(target=fake_worker, daemon=True)
        worker.start()

        def caller(method: str) -> None:
            executor.collective_rpc(method, unique_reply_rank=0, timeout=5.0)

        t1 = threading.Thread(target=caller, args=("sleep",), daemon=True)
        t2 = threading.Thread(target=caller, args=("sleep",), daemon=True)
        t1.start()
        _time.sleep(0.01)  # let t1 acquire _serialize_lock first
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        worker.join(timeout=5)

        assert len(enqueue_times) == 2, "both sleep calls must reach broadcast"
        delta = enqueue_times[1] - enqueue_times[0]
        assert delta >= 0.08, (
            f"non-interleavable methods must serialise; "
            f"second broadcast was only {delta * 1000:.1f}ms after the first "
            f"(expected >=80ms because the worker holds for 100ms)"
        )

    def test_benchmark_smoke_overhead_under_budget(self) -> None:
        """200 sequential no-op RPCs through the demux loop add a
        median latency well below the 5ms smoke ceiling. The real
        budget assertion (<50µs added vs no-dispatcher baseline) lives
        in ``benchmarks/diffusion/bench_executor_demux.py``."""
        import time as _time

        executor, req_q, res_q = _make_executor()

        def fake_worker(n: int) -> None:
            for _ in range(n):
                req = req_q.get(timeout=10)
                cid = req["correlation_id"]
                res_q.put(
                    {
                        "type": "rpc_reply",
                        "correlation_id": cid,
                        "payload": _tagged_output("ok"),
                    }
                )

        n = 200
        worker = threading.Thread(target=fake_worker, args=(n,), daemon=True)
        worker.start()

        latencies: list[float] = []
        for _ in range(n):
            t0 = _time.perf_counter()
            executor.collective_rpc("ping", unique_reply_rank=0, timeout=5.0)
            latencies.append(_time.perf_counter() - t0)
        worker.join(timeout=10)

        latencies.sort()
        median = latencies[len(latencies) // 2]
        # Smoke ceiling: 5ms per call. The real budget (<50µs delta vs
        # baseline) is enforced by the standalone benchmark.
        assert median < 0.005, f"median per-call latency {median * 1e6:.1f} µs exceeds 5ms smoke ceiling"
