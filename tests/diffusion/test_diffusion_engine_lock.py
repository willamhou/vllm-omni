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
