# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-level regression tests for DiffusionEngine.

These tests verify naming conventions and patterns by inspecting source code
directly. They are intentionally coupled to the source layout and should be
updated whenever the metrics construction code is refactored.
"""

from __future__ import annotations

import ast
import os

_ENGINE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "vllm_omni",
        "diffusion",
        "diffusion_engine.py",
    )
)


def _read_engine_source() -> str:
    with open(_ENGINE_PATH) as f:
        return f.read()


class TestMetricKeys:
    """Verify metric naming conventions in DiffusionEngine.step() output."""

    def test_no_duplicate_preprocess_key(self) -> None:
        """The metrics dict should not contain both
        'preprocess_time_ms' and 'preprocessing_time_ms'."""
        source = _read_engine_source()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "step":
                step_source = ast.get_source_segment(source, node)
                assert step_source is not None
                assert "preprocessing_time_ms" not in step_source, (
                    "Found duplicate key 'preprocessing_time_ms' in step() — should only use 'preprocess_time_ms'"
                )
                break

    def test_metric_key_naming_consistency(self) -> None:
        """exec_time should measure execution only, total_time should
        measure the full step including pre/post processing."""
        source = _read_engine_source()
        lines = source.split("\n")

        found_exec = False
        found_total = False
        for line in lines:
            if '"diffusion_engine_exec_time_ms"' in line:
                found_exec = True
                assert "exec_total_time" in line, (
                    "diffusion_engine_exec_time_ms should measure executor time only (exec_total_time)"
                )
            if '"diffusion_engine_total_time_ms"' in line:
                found_total = True
                assert "step_total_ms" in line, (
                    "diffusion_engine_total_time_ms should measure full step time (step_total_ms)"
                )
        assert found_exec, "diffusion_engine_exec_time_ms key not found in source"
        assert found_total, "diffusion_engine_total_time_ms key not found in source"


class TestDummyRunAllocation:
    """Verify _dummy_run generates exact-sized audio arrays."""

    def test_no_oversized_allocation(self) -> None:
        """_dummy_run should not allocate more audio than needed."""
        source = _read_engine_source()
        assert "audio_sr * audio_duration_sec" not in source, (
            "_dummy_run should generate exact-sized audio, not allocate and slice"
        )
