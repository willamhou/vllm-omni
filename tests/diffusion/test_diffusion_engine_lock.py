# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DiffusionEngine lock narrowing with Condition-based execution guard.

Verifies:
1. collective_rpc is not blocked during GPU execution
2. A second add_req_and_wait_for_response does NOT duplicate execution
3. Scheduler operations remain serialized
"""

from __future__ import annotations

import threading
import time

import pytest


class TestConditionBasedLock:
    """Test the Condition + _executing flag pattern."""

    def test_rpc_not_blocked_during_execution(self) -> None:
        """collective_rpc should acquire _rpc_lock while execute_fn runs."""
        lock = threading.RLock()
        cond = threading.Condition(lock)
        executing = False
        execute_started = threading.Event()
        rpc_acquired = threading.Event()

        def simulate_request():
            nonlocal executing
            with cond:
                executing = True
            # Lock released — simulate GPU work
            execute_started.set()
            time.sleep(0.2)
            with cond:
                executing = False
                cond.notify_all()

        def simulate_rpc():
            execute_started.wait(timeout=2.0)
            time.sleep(0.05)
            acquired = lock.acquire(timeout=0.1)
            if acquired:
                rpc_acquired.set()
                lock.release()

        t1 = threading.Thread(target=simulate_request)
        t2 = threading.Thread(target=simulate_rpc)
        t1.start()
        t2.start()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        assert rpc_acquired.is_set(), "collective_rpc should acquire lock during execute_fn"

    def test_second_caller_waits_no_duplicate_execution(self) -> None:
        """A second add_req_and_wait_for_response caller must wait on
        _executing flag, not schedule a duplicate execution."""
        lock = threading.RLock()
        cond = threading.Condition(lock)
        executing = False
        exec_count = 0
        exec_started = threading.Event()

        def simulate_caller():
            nonlocal executing, exec_count
            with cond:
                while executing:
                    cond.wait()
                # schedule() — would get a request
                executing = True

            # execute_fn — GPU work
            exec_count += 1
            exec_started.set()
            time.sleep(0.15)

            with cond:
                executing = False
                cond.notify_all()

        t1 = threading.Thread(target=simulate_caller)
        t2 = threading.Thread(target=simulate_caller)
        t1.start()
        time.sleep(0.02)  # ensure t1 grabs the flag first
        t2.start()
        t1.join(timeout=3.0)
        t2.join(timeout=3.0)

        # Both should execute, but sequentially (not concurrently)
        assert exec_count == 2, f"Expected 2 sequential executions, got {exec_count}"

    def test_no_concurrent_execution(self) -> None:
        """Two callers must never execute simultaneously."""
        lock = threading.RLock()
        cond = threading.Condition(lock)
        executing = False
        concurrent_detected = threading.Event()
        active_count = 0
        active_lock = threading.Lock()

        def simulate_caller():
            nonlocal executing, active_count
            with cond:
                while executing:
                    cond.wait()
                executing = True

            # execute_fn
            with active_lock:
                active_count += 1
                if active_count > 1:
                    concurrent_detected.set()
            time.sleep(0.1)
            with active_lock:
                active_count -= 1

            with cond:
                executing = False
                cond.notify_all()

        threads = [threading.Thread(target=simulate_caller) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not concurrent_detected.is_set(), "Concurrent execution detected — _executing flag failed"

    def test_engine_dead_error_clears_executing_flag(self) -> None:
        """If execute_fn raises an EngineDeadError-like exception, the
        executing flag must be cleared and waiters notified before the
        exception propagates. Otherwise a second caller waiting on
        cond.wait() would deadlock forever."""

        class FakeEngineDeadError(Exception):
            pass

        lock = threading.RLock()
        cond = threading.Condition(lock)
        executing = False
        waiter_unblocked = threading.Event()
        captured: list[BaseException] = []

        def dying_caller():
            nonlocal executing
            with cond:
                while executing:
                    cond.wait()
                executing = True
            try:
                # execute_fn surrogate that signals engine death
                raise FakeEngineDeadError("engine died mid-flight")
            except FakeEngineDeadError:
                with cond:
                    executing = False
                    cond.notify_all()
                raise

        def waiter():
            nonlocal executing
            with cond:
                while executing:
                    cond.wait()
            waiter_unblocked.set()

        def run_dying():
            try:
                dying_caller()
            except FakeEngineDeadError as exc:
                captured.append(exc)

        t1 = threading.Thread(target=run_dying)
        t1.start()
        time.sleep(0.02)  # ensure t1 sets executing=True first
        t2 = threading.Thread(target=waiter)
        t2.start()

        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        assert captured, "FakeEngineDeadError should propagate to the caller"
        assert waiter_unblocked.is_set(), (
            "Waiter must be unblocked after engine dies; otherwise the "
            "_executing flag deadlocks all subsequent callers"
        )

    def test_source_engine_dead_clears_executing(self) -> None:
        """AST check: the EngineDeadError handler in
        add_req_and_wait_for_response must clear _executing and call
        notify_all() before re-raising, otherwise the rebased PR
        introduces a deadlock when engine dies mid-execution."""
        import ast
        import os

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
        with open(engine_path) as f:
            source = f.read()

        tree = ast.parse(source)
        target_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiffusionEngine":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "add_req_and_wait_for_response":
                        target_func = item
                        break
        assert target_func is not None, "add_req_and_wait_for_response not found"

        for handler in ast.walk(target_func):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            exc_type = handler.type
            if not (isinstance(exc_type, ast.Name) and exc_type.id == "EngineDeadError"):
                continue
            handler_src = ast.get_source_segment(source, handler) or ""
            assert "_executing = False" in handler_src, (
                "EngineDeadError handler must clear _executing flag before re-raising"
            )
            assert "notify_all" in handler_src, (
                "EngineDeadError handler must notify_all() before re-raising"
            )
            return
        pytest.fail("EngineDeadError handler missing in add_req_and_wait_for_response")

    def test_integration_real_path_no_duplicate_execute(self) -> None:
        """Integration test invoking the real add_req_and_wait_for_response
        on a DiffusionEngine instance (constructed via __new__ to bypass
        the heavy __init__) with a fake scheduler and sleep-based
        execute_fn. Two concurrent callers must not invoke execute_fn
        while another invocation is still in flight, and execute_fn must
        run exactly once per request.

        This addresses reviewer feedback that the previous tests only
        verified the locking pattern in isolation and never exercised
        the real method against a scheduler that re-yields running
        requests."""
        from types import SimpleNamespace

        from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

        class FakeScheduler:
            def __init__(self) -> None:
                self._counter = 0
                self._waiting: list[str] = []
                self._running: list[str] = []
                self._lock = threading.Lock()

            def add_request(self, request: object) -> str:
                with self._lock:
                    self._counter += 1
                    req_id = f"req_{self._counter}"
                    self._waiting.append(req_id)
                    return req_id

            def schedule(self) -> SimpleNamespace:
                with self._lock:
                    # Re-yield in-flight running request if any (this is
                    # exactly the behaviour reviewer warned about).
                    if not self._running and self._waiting:
                        self._running.append(self._waiting.pop(0))
                    if self._running:
                        return SimpleNamespace(
                            is_empty=False,
                            scheduled_req_ids=[self._running[0]],
                            finished_req_ids=[],
                        )
                    return SimpleNamespace(
                        is_empty=True,
                        scheduled_req_ids=[],
                        finished_req_ids=[],
                    )

            def update_from_output(self, sched_output: SimpleNamespace, runner_output: SimpleNamespace) -> list[str]:
                with self._lock:
                    req_id = sched_output.scheduled_req_ids[0]
                    if req_id in self._running:
                        self._running.remove(req_id)
                    return [req_id]

            def has_requests(self) -> bool:
                with self._lock:
                    return bool(self._waiting or self._running)

            def get_request_state(self, sched_req_id: str) -> SimpleNamespace:
                return SimpleNamespace()

            def pop_request_state(self, sched_req_id: str) -> SimpleNamespace:
                return SimpleNamespace()

            def get_sched_req_id(self, request_id: str) -> None:
                return None

        active = {"count": 0, "max": 0, "calls": 0}
        active_lock = threading.Lock()

        def slow_execute(sched_output: SimpleNamespace) -> SimpleNamespace:
            with active_lock:
                active["count"] += 1
                active["calls"] += 1
                if active["count"] > active["max"]:
                    active["max"] = active["count"]
            try:
                time.sleep(0.15)
            finally:
                with active_lock:
                    active["count"] -= 1
            return SimpleNamespace(
                req_id=sched_output.scheduled_req_ids[0],
                step_index=None,
                finished=True,
                result=SimpleNamespace(),
            )

        engine = object.__new__(DiffusionEngine)
        engine._rpc_lock = threading.RLock()
        engine._exec_cond = threading.Condition(engine._rpc_lock)
        engine._executing = False
        engine.scheduler = FakeScheduler()
        engine.execute_fn = slow_execute
        engine._process_aborts_queue = lambda: None
        engine._finalize_finished_request = lambda sched_req_id, **kw: SimpleNamespace(req_id=sched_req_id)

        fake_request = SimpleNamespace()
        results: list[object] = []
        results_lock = threading.Lock()

        def caller() -> None:
            out = engine.add_req_and_wait_for_response(fake_request)
            with results_lock:
                results.append(out)

        t1 = threading.Thread(target=caller)
        t2 = threading.Thread(target=caller)
        t1.start()
        time.sleep(0.01)
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not t1.is_alive() and not t2.is_alive(), "Threads deadlocked"
        assert active["calls"] == 2, f"Expected execute_fn called twice, got {active['calls']}"
        assert active["max"] == 1, f"Concurrent execute_fn detected (max in flight={active['max']})"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    def test_source_uses_exec_cond(self) -> None:
        """Verify add_req_and_wait_for_response uses _exec_cond (Condition)
        and _executing flag, not a single wrapping _rpc_lock."""
        import ast
        import os

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
        with open(engine_path) as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DiffusionEngine":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "add_req_and_wait_for_response":
                        func_src = ast.get_source_segment(source, item)
                        assert func_src is not None
                        assert "_exec_cond" in func_src, "Should use _exec_cond (Condition) for lock management"
                        assert "_executing" in func_src, "Should use _executing flag to prevent duplicate execution"
                        # Should have multiple `with self._exec_cond:` blocks (narrowed scope)
                        with_cond_count = 0
                        for child in ast.walk(item):
                            if isinstance(child, ast.With):
                                for wi in child.items:
                                    ctx = wi.context_expr
                                    if isinstance(ctx, ast.Attribute) and ctx.attr == "_exec_cond":
                                        with_cond_count += 1
                        assert with_cond_count >= 3, (
                            f"Expected >= 3 _exec_cond acquisitions (narrowed scope), found {with_cond_count}"
                        )
                        return
        pytest.fail("add_req_and_wait_for_response not found")
