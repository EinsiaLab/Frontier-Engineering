from __future__ import annotations

import unittest

from frontier_eval.algorithms.shinkaevolve import shinkaevolve_entrypoint as entrypoint


class TestShinkaEvolveEntrypoint(unittest.TestCase):
    def test_extract_metrics_and_artifacts_from_nested_dict(self) -> None:
        metrics, artifacts = entrypoint._extract_metrics_and_artifacts(
            {
                "metrics": {"combined_score": -1.0, "valid": 0.0},
                "artifacts": {"error_message": "boom"},
            }
        )

        self.assertEqual(metrics["combined_score"], -1.0)
        self.assertEqual(artifacts["error_message"], "boom")

    def test_primary_error_message_reads_nested_unified_artifact(self) -> None:
        error = entrypoint._primary_error_message(
            {
                "user_artifact::error_message": "candidate infeasible on case 2",
            }
        )

        self.assertEqual(error, "candidate infeasible on case 2")

    def test_synthesize_text_feedback_prioritizes_error_and_runtime_problem(self) -> None:
        feedback = entrypoint._synthesize_text_feedback(
            {"combined_score": -1e18, "valid": 0.0, "benchmark_returncode": 0.0},
            {
                "user_artifact::error_message": "Traceback: KeyError: 'wind_u'",
                "agent_files": "\n".join(
                    [
                        "Task.md",
                        "README.md",
                        "baseline/solution.py",
                        "runtime/problem.py",
                    ]
                ),
                "constraints": "Edit only scripts/init.py.",
                "agent_file::Task.md": "Task contract",
                "agent_file::README.md": "README content",
                "agent_file::baseline/solution.py": "def solve(instance): return baseline()",
                "agent_file::runtime/problem.py": "INSTANCE_KEYS = ['time_grid', 'weather_cube']",
                "benchmark_stdout": '{"combined_score": -1e18, "valid": 0.0}',
            },
        )

        self.assertIn("## Error Message", feedback)
        self.assertIn("KeyError: 'wind_u'", feedback)
        self.assertIn("## Constraints", feedback)
        self.assertIn("## Agent File: runtime/problem.py", feedback)
        self.assertIn("INSTANCE_KEYS", feedback)
        self.assertLess(feedback.index("## Error Message"), feedback.index("## Constraints"))
        self.assertLess(feedback.index("## Benchmark Stdout"), feedback.index("## Agent File: runtime/problem.py"))

    def test_synthesize_text_feedback_includes_program_stderr_and_traceback(self) -> None:
        feedback = entrypoint._synthesize_text_feedback(
            {"combined_score": 0.0, "valid": 0.0},
            {
                "error_message": "candidate program exited non-zero",
                "program_stderr": "Traceback (most recent call last):\nValueError: bad input",
                "traceback": "ValueError: bad input\n  at solver.py:13",
            },
        )

        self.assertIn("## Error Message", feedback)
        self.assertIn("candidate program exited non-zero", feedback)
        self.assertIn("## Program Stderr", feedback)
        self.assertIn("ValueError: bad input", feedback)
        self.assertIn("## Traceback", feedback)

    def test_context_bundle_prefers_full_stderr_and_builds_log_bridges(self) -> None:
        bundle = entrypoint._build_context_bundle(
            {"combined_score": 0.0, "valid": 0.0},
            {
                "error_message": "build failed",
                "make_stderr": "short stderr",
                "make_stderr_full": "full stderr details",
                "make_stdout": "stdout details",
            },
        )

        self.assertFalse(bundle.correct)
        self.assertEqual(bundle.primary_error, "build failed")
        self.assertIn("full stderr details", bundle.text_feedback)
        self.assertNotIn("short stderr", bundle.text_feedback)
        self.assertIn("full stderr details", bundle.stderr_bridge)
        self.assertIn("stdout details", bundle.stdout_bridge)
        self.assertIn("make_stderr_full", bundle.selected_keys)

    def test_context_bundle_reports_omitted_keys_when_budget_is_small(self) -> None:
        bundle = entrypoint._build_context_bundle(
            {"combined_score": 0.0, "valid": 0.0},
            {
                "error_message": "boom",
                "program_stderr": "stderr",
                "benchmark_stdout": "stdout",
                "constraints": "keep interface stable",
                "extra_context_1": "x1",
                "extra_context_2": "x2",
                "extra_context_3": "x3",
            },
            text_feedback_max_chars=500,
        )

        self.assertIn("## Omitted Context", bundle.text_feedback)
        self.assertTrue(bundle.omitted_keys)


if __name__ == "__main__":
    unittest.main()
