"""STAND-IN for beyondbench/core/base_task.py (BeyondBench 0.2.1, Apache-2.0). NOT the official file.

The official BaseTask (1,080 lines) imports transformers / tiktoken at import time, owns the model handler, the generation loop and the fold statistics. titan_eval does the
generation itself (vLLM through lighteval) and re-implements the fold statistics in lighteval.tasks.tasks.beyondbench (`score_item`, the metric), so the 44 task classes only need the
constructor attributes that their `generate_data` / `create_prompt` / `evaluate_response` read: `seed`, `min_val`, `max_val`, `num_samples`, `num_folds` (and the abstract members).
`tests/unit/tasks/test_beyondbench.py` proves that the two give the same items and verdicts (the stored prompts / responses / verdicts of the owner's two official runs).
"""

from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(self, min_val, max_val, num_folds, num_samples, seed=None, **kwargs):
        self.min_val = min_val
        self.max_val = max_val
        self.num_folds = num_folds
        self.num_samples = num_samples
        self.seed = seed

    @property
    @abstractmethod
    def task_name(self):
        """Return task name, to be implemented by subclasses"""

    @abstractmethod
    def generate_data(self, **kwargs):
        """Generate evaluation data, to be implemented by subclasses"""

    @abstractmethod
    def create_prompt(self, data_point):
        """Create prompt for the task, to be implemented by subclasses"""

    @abstractmethod
    def evaluate_response(self, response, data_point):
        """Evaluate model response, to be implemented by subclasses"""
