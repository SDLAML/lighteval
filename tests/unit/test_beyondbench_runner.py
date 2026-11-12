"""CPU-only checks for the BeyondBench evaluation deadline."""

import importlib.util
import json
import re
import signal
import tempfile
import unittest
from pathlib import Path


_RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "beyondbench_eval.py"
_SPEC = importlib.util.spec_from_file_location("beyondbench_eval", _RUNNER_PATH)
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


class TestBeyondBenchRunner(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.task_dir = Path(directory.name)
        task_dir = self.task_dir

        class Task:
            task_name = "sum"

            def evaluate_response(self, response, data_point):
                if response.startswith("slow"):
                    # Mimic parsers that swallow ordinary exceptions internally.
                    try:
                        re.search(r"(.+?)\s+is the answer", " " * 16000 + "x")
                    except Exception:
                        return {"swallowed": True}
                return {"accuracy": int(response) == sum(data_point)}

        Task.task_dir = task_dir

        class Registry:
            def get_task_class(self, task_name):
                return Task if task_name == "sum" else None

        self.registry_class = Registry

    def test_deadline_records_response_and_allows_next_sample(self):
        runner.install_evaluation_timeout(self.registry_class, 0.05)
        task = self.registry_class().get_task_class("sum")()
        previous_handler = signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError, "Response evaluation exceeded"):
            task.evaluate_response("slow response", [1, 2])

        records = [json.loads(line) for line in (self.task_dir / "evaluation_timeouts.jsonl").read_text().splitlines()]
        self.assertEqual(records[0]["data_point"], [1, 2])
        self.assertEqual(records[0]["model_response"], "slow response")
        self.assertIn("re.search", records[0]["traceback"])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous_handler)
        self.assertEqual(task.evaluate_response("3", [1, 2]), {"accuracy": True})

    def test_registry_handles_repeated_and_missing_lookups(self):
        runner.install_evaluation_timeout(self.registry_class, 1)
        registry = self.registry_class()
        evaluate = registry.get_task_class("sum").evaluate_response
        self.assertIs(registry.get_task_class("sum").evaluate_response, evaluate)
        self.assertIsNone(registry.get_task_class("missing"))

    def test_ordinary_error_restores_signal_state(self):
        runner.install_evaluation_timeout(self.registry_class, 1)
        task = self.registry_class().get_task_class("sum")()
        previous_handler = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(ValueError):
            task.evaluate_response("invalid", [1, 2])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous_handler)
        self.assertFalse((self.task_dir / "evaluation_timeouts.jsonl").exists())

    def test_invalid_deadline(self):
        for seconds in [0, -1, float("inf"), float("nan")]:
            with self.subTest(seconds=seconds):
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    runner.install_evaluation_timeout(self.registry_class, seconds)


if __name__ == "__main__":
    unittest.main()
