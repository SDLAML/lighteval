# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

import unittest
from unittest.mock import Mock, patch

from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.tasks.registry import Registry


class TestOfflineTaskRegistration(unittest.TestCase):
    def test_local_judge_initialization_does_not_query_hub(self):
        with (
            patch("huggingface_hub.HfApi.list_models", side_effect=AssertionError("Unexpected Hub search")),
            patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("Unexpected Hub lookup")),
        ):
            for backend in ("transformers", "vllm"):
                with self.subTest(backend=backend):
                    metric = JudgeLLM(
                        judge_model_name="/local/checkpoints/judge",
                        template=Mock(),
                        process_judge_response=Mock(),
                        judge_backend=backend,
                    )
                    self.assertEqual(metric.judge.model, "/local/checkpoints/judge")
                    self.assertIsNone(metric.judge.pipe)

    def test_truthfulqa_and_popqa_register_without_multilingual_or_hub_queries(self):
        with (
            patch("huggingface_hub.HfApi.list_models", side_effect=AssertionError("Unexpected Hub search")),
            patch("huggingface_hub.HfApi.model_info", side_effect=AssertionError("Unexpected Hub lookup")),
        ):
            registry = Registry(tasks="truthfulqa:gen|5,popqa:gen|5", load_multilingual=False)
            tasks = registry.load_tasks()
        self.assertEqual(set(tasks), {"truthfulqa:gen|5", "popqa:gen|5"})


if __name__ == "__main__":
    unittest.main()
