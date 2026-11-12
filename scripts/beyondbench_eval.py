"""Run BeyondBench with a time limit on each response evaluation.

BeyondBench 0.2.1 can spend unbounded time in answer-extraction regexes.
BEYONDBENCH_EVAL_TIMEOUT controls the limit in seconds (default: 10).
Timed-out responses are saved for inspection and reported as parsing failures
by BeyondBench's existing evaluation loop. Inference is not time-limited.
"""

import functools
import json
import math
import os
import signal
import time
import traceback
from pathlib import Path


class _EvaluationExpired(BaseException):
    # Some parsers catch Exception internally; the deadline must escape them.
    pass


def limit_evaluation(evaluate_response, seconds):
    @functools.wraps(evaluate_response)
    def evaluate(self, response, data_point):
        def expired(signum, frame):
            raise _EvaluationExpired("".join(traceback.format_stack(frame)))

        started = time.monotonic()
        previous_handler = signal.signal(signal.SIGALRM, expired)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            try:
                return evaluate_response(self, response, data_point)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_timer[0] > 0:
                    remaining = max(0.000001, previous_timer[0] - (time.monotonic() - started))
                    signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])
        except _EvaluationExpired as error:
            diagnostic_path = Path(self.task_dir) / "evaluation_timeouts.jsonl"
            diagnostic = {
                "task": self.task_name,
                "timeout_seconds": seconds,
                "data_point": data_point,
                "model_response": response,
                "traceback": str(error),
            }
            with diagnostic_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(diagnostic, default=str) + "\n")
            # BaseTask records this as parsing_success=False, accuracy=0.
            raise TimeoutError(
                f"Response evaluation exceeded {seconds:g}s; saved to {diagnostic_path}"
            ) from None

    evaluate._beyondbench_time_limited = True
    return evaluate


def install_evaluation_timeout(registry_class, seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("BEYONDBENCH_EVAL_TIMEOUT must be a finite positive number")

    get_task_class = registry_class.get_task_class

    @functools.wraps(get_task_class)
    def get_bounded_task_class(self, task_name):
        task_class = get_task_class(self, task_name)
        if task_class is not None and not getattr(task_class.evaluate_response, "_beyondbench_time_limited", False):
            task_class.evaluate_response = limit_evaluation(task_class.evaluate_response, seconds)
        return task_class

    registry_class.get_task_class = get_bounded_task_class


def main():
    seconds = float(os.environ.get("BEYONDBENCH_EVAL_TIMEOUT", "10"))
    from beyondbench.cli.main import main as beyondbench_main
    from beyondbench.core.task_registry import TaskRegistry

    install_evaluation_timeout(TaskRegistry, seconds)
    beyondbench_main()


if __name__ == "__main__":
    main()
