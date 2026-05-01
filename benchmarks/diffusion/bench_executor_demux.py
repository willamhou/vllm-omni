#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MultiprocDiffusionExecutor sender-driven demux overhead.

Compares two paths over a no-op RPC:

* **baseline** — direct ``_broadcast_mq.enqueue`` + ``_result_mq.dequeue``
  with no correlation_id, no _pending registration. This is the lower
  bound any envelope-based scheme can achieve through the same fixture.
* **dispatcher** — full ``collective_rpc`` path with cid stamping, the
  sender-driven demux loop, _pending bookkeeping, and inbox lookups.

Reports min / p50 / p90 / p99 / max for each, plus the median delta.
The Step A budget is **median delta < 50 µs**; the script exits non-zero
on regression so it can be wired into CI.

Usage::

    python benchmarks/diffusion/bench_executor_demux.py [--iterations N]
"""

from __future__ import annotations

import argparse
import queue
import statistics
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor


def _tagged_output(tag: str) -> DiffusionOutput:
    return DiffusionOutput(output=torch.tensor([0]), error=tag)


def _build_executor() -> tuple[MultiprocDiffusionExecutor, queue.Queue, queue.Queue]:
    od_cfg = SimpleNamespace(num_gpus=1)
    # Bypass _init_executor so we don't spawn workers.
    real_init = MultiprocDiffusionExecutor._init_executor
    MultiprocDiffusionExecutor._init_executor = lambda self: None
    try:
        executor = MultiprocDiffusionExecutor(od_cfg)
    finally:
        MultiprocDiffusionExecutor._init_executor = real_init

    req_q: queue.Queue = queue.Queue()
    res_q: queue.Queue = queue.Queue()
    executor._broadcast_mq = SimpleNamespace(enqueue=req_q.put)

    def _fake_dequeue(timeout=None):
        try:
            return res_q.get(timeout=timeout if timeout is not None else 10)
        except queue.Empty as e:
            raise TimeoutError() from e

    executor._result_mq = SimpleNamespace(dequeue=_fake_dequeue)
    executor._closed = False
    executor._processes = []
    executor.is_failed = False
    executor._failure_callbacks = []
    return executor, req_q, res_q


def _start_echo_worker(req_q: queue.Queue, res_q: queue.Queue, n: int) -> threading.Thread:
    """Echo each request as either a tagged envelope (if cid present)
    or a bare DiffusionOutput (if not), matching the production worker."""

    def _run() -> None:
        for _ in range(n):
            req = req_q.get(timeout=10)
            payload = _tagged_output("ok")
            cid = req.get("correlation_id") if isinstance(req, dict) else None
            if cid is None:
                res_q.put(payload)
            else:
                res_q.put({"type": "rpc_reply", "correlation_id": cid, "payload": payload})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _measure(label: str, op: Callable[[], None], n: int) -> list[float]:
    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        op()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples


def _percentile(sorted_samples: list[float], p: float) -> float:
    idx = int(len(sorted_samples) * p)
    return sorted_samples[min(idx, len(sorted_samples) - 1)]


def _print_stats(label: str, samples: list[float]) -> None:
    print(
        f"  {label:11s} min={_percentile(samples, 0.0) * 1e6:7.1f} µs"
        f"  p50={statistics.median(samples) * 1e6:7.1f} µs"
        f"  p90={_percentile(samples, 0.9) * 1e6:7.1f} µs"
        f"  p99={_percentile(samples, 0.99) * 1e6:7.1f} µs"
        f"  max={_percentile(samples, 1.0) * 1e6:7.1f} µs"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bench Step A demux overhead")
    parser.add_argument("--iterations", type=int, default=10000, help="iterations per config")
    parser.add_argument("--budget-us", type=float, default=50.0, help="median delta budget (µs)")
    args = parser.parse_args()
    n = args.iterations

    print(f"iterations={n}, budget={args.budget_us:.1f} µs median delta")
    print()

    # ── baseline: direct mq, no executor.collective_rpc ───────────────
    executor_b, req_qb, res_qb = _build_executor()
    worker_b = _start_echo_worker(req_qb, res_qb, n)

    def _baseline_op() -> None:
        executor_b._broadcast_mq.enqueue({"type": "rpc", "method": "ping"})
        executor_b._result_mq.dequeue(timeout=10)

    baseline = _measure("baseline", _baseline_op, n)
    worker_b.join(timeout=10)

    # ── dispatcher: full Step A path ──────────────────────────────────
    executor_d, req_qd, res_qd = _build_executor()
    worker_d = _start_echo_worker(req_qd, res_qd, n)

    def _dispatch_op() -> None:
        executor_d.collective_rpc("ping", unique_reply_rank=0)

    dispatcher = _measure("dispatcher", _dispatch_op, n)
    worker_d.join(timeout=10)

    print("Per-call latency:")
    _print_stats("baseline", baseline)
    _print_stats("dispatcher", dispatcher)
    print()

    delta = statistics.median(dispatcher) - statistics.median(baseline)
    delta_us = delta * 1e6
    verdict = "PASS" if delta_us <= args.budget_us else "FAIL"
    print(f"median delta: {delta_us:+.1f} µs  (budget {args.budget_us:.1f} µs) -> {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
