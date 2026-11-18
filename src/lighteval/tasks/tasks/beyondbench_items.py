"""BeyondBench easy suite (BASE): the items and the scorer. Not a task module (no TASKS_TABLE; `beyondbench.py` is the task) and no heavy imports, so the data preparation
and the tests start in a second.

What is BeyondBench: a generated reasoning benchmark (arXiv 2509.24210, ICLR 2026; github.com/ctrl-gaurav/BeyondBench, PyPI `beyondbench`, Apache-2.0). Its EASY suite is 44 list /
arithmetic tasks (sum, sorting, median, set intersection, ...). Every item is a prompt that asks for the answer as `\\boxed{...}`; the gold is computed from the generated numbers.

The protocol is the OWNER's (scripts/run_beyondbench.sh + scripts/beyondbench_eval.py of the fork, official package 0.2.1, owner 11-12 / 11-18): suite easy, 100 items per (task, list size),
seed 42, the list sizes and number ranges per task group below, greedy, top_p 1, up to 4,096 new tokens, no chat template, no stop string, the official scorer with a 10 s deadline per answer.
The items come from the OFFICIAL generators (`beyondbench_official/tasks/easy/*_task.py`, byte-identical vendored files; `generate_data` re-seeds `random` with the seed at every call, so an
item set depends on (task, list size, range, seed) only, never on the order or on a model). 8,400 items = 84 (task, size) configurations x 100.

Scoring = the official `evaluate_response` of each task class (the same function the package's BaseTask calls) with the official fold bookkeeping (BaseTask._run_fold_batch / process_fold_metrics):
 - a reply that is the empty string is a failed generation: accuracy 0, and the OFFICIAL accuracy leaves it out of the denominator (`acc_official`); titan_eval's `acc` keeps it in (a model that
   answers nothing did not answer correctly);
 - any exception inside evaluate_response (also the 10 s deadline of the owner's wrapper) = accuracy 0, a parsing failure, kept in the denominator;
 - `instruction_followed` = the parsers found an answer.
Two documented differences of titan_eval's primary `acc` from the package's number (both reported next to it as `acc_official`, which is the package's own accuracy):
 - empty replies stay in the denominator (above);
 - `second_minimum`: the official evaluate_response reads `parse_number(response)` = the FIRST number anywhere in the reply, not the boxed answer (all 43 other tasks go through the boxed-answer
   parsers); stored official runs score it 0.04-0.08 against 0.75-0.77 for its sibling second_maximum. `acc` uses the boxed-answer parser (`parse_arithmetic_result`, the one every sibling
   task uses) and the same comparison; `acc_official` keeps the official verdict.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import signal
import threading
import time
from pathlib import Path

OFFICIAL_VERSION = "0.2.1"
SEED = 42
DATAPOINTS = 100
MAX_NEW_TOKENS = 4096
EVAL_TIMEOUT_SECONDS = 10.0                  # scripts/beyondbench_eval.py: BEYONDBENCH_EVAL_TIMEOUT default

GENERAL_TASKS = (
    "sorting", "sum", "odd_count", "even_count", "find_maximum", "find_minimum", "mean", "mode", "second_maximum", "range", "index_of_maximum", "count_negative", "count_unique",
    "max_adjacent_difference", "count_greater_than_previous", "sum_of_max_indices", "count_palindromic", "longest_increasing_subsequence", "sum_of_digits", "count_perfect_squares",
    "alternating_sum", "count_multiples", "local_maxima_count", "weighted_sum", "running_average", "parity_check", "cumulative_sum", "reverse_list", "rotate_list", "moving_average",
    "second_minimum", "variance", "standard_deviation",
)
# (group, list sizes, (range_min, range_max), tasks): the groups of scripts/run_beyondbench.sh, in its order (the script's comments give the reasons: median 9 / 16 = odd and even length, element
# frequency range -10..10 = repeated values, multiplication 2 / 4 factors = products inside the exact range of the official float scorer, set operations range -20..20 = overlap)
GROUPS = (
    ("general", (8, 16), (-100, 100), GENERAL_TASKS),
    ("median", (9, 16), (-100, 100), ("median",)),
    ("element_frequency", (8, 16), (-10, 10), ("element_frequency",)),
    ("multiplication", (2, 4), (-100, 100), ("multiplication",)),
    ("paired_lists", (5, 10), (-100, 100), ("interleave_lists", "dot_product")),
    ("set_operations", (6, 12), (-20, 20), ("set_intersection", "set_difference")),
    ("fixed_inputs", (2,), (-100, 100), ("absolute_difference", "comparison", "division", "subtraction")),
)
TASKS = tuple(t for _, _, _, tasks in GROUPS for t in tasks)                      # the 44 easy tasks, in the script's order
NO_LIST_SIZE = ("comparison", "division", "subtraction")                          # their generate_data() takes no size (the script's `--list-sizes 2` is ignored by them)
N_ITEMS = 8400                                                                    # 33 x 2 + 4 x 2 + ... = 84 configurations x 100 (the owner's 2 stored official runs: 84 x 100)
ITEMS_FILE = "beyondbench_easy.jsonl"
ITEMS_FOLDER = "beyondbench_easy_v021"
# sha256 of the prepared file (canonical JSON, one row per line): the generation is deterministic, `prepare` writes this file and `load_items` checks it at every load
ITEMS_SHA256 = "ceac3b0356826bb92f3e5e84ac6ef613126cca94b382d9bd9bd0d5aabee2d4aa"

OFFICIAL_PACKAGE = "lighteval.tasks.tasks.beyondbench_official"


def group_of(task: str) -> str:
    for g, _, _, tasks in GROUPS:
        if task in tasks:
            return g
    raise KeyError(task)


def configs():
    """[(group, task, list_size | None, range_min, range_max)] of the 84 configurations, in the script's order (sizes ascending)."""
    out = []
    for group, sizes, (lo, hi), tasks in GROUPS:
        for task in tasks:
            if task in NO_LIST_SIZE:
                out.append((group, task, None, lo, hi))
            else:
                out.extend((group, task, s, lo, hi) for s in sizes)
    return out


_CLASSES: dict[str, type] = {}


def task_class(task: str) -> type:
    """The official task class of an easy task (imported from the vendored package)."""
    if task not in _CLASSES:
        mod = importlib.import_module(f"{OFFICIAL_PACKAGE}.tasks.easy.{task}_task")
        found = [v for k, v in vars(mod).items() if isinstance(v, type) and k.endswith("Task") and k != "BaseTask" and v.__module__ == mod.__name__]
        if len(found) != 1:
            raise RuntimeError(f"{mod.__name__}: expected one task class, found {found}")
        _CLASSES[task] = found[0]
    return _CLASSES[task]


def make_task(task: str, lo: int = -100, hi: int = 100, seed: int | None = SEED, num_samples: int = DATAPOINTS):
    return task_class(task)(min_val=lo, max_val=hi, num_folds=1, num_samples=num_samples, seed=seed)


_GOLD_KEYS = ("ground_truth", "sum", "expected_relation")       # `sum` names its gold `sum`, `comparison` `expected_relation`; every other easy task `ground_truth`
_PARSED_KEYS = ("parsed_answer", "predicted_answer", "calculated_answer", "parsed_min", "parsed", "predicted")


def gold_of(inst, data_point):
    """The gold the official evaluate_response computes for this data point (called with an empty reply, which every task scores 0 without an error)."""
    res = inst.evaluate_response("", data_point)
    for k in _GOLD_KEYS:
        if k in res:
            return res[k]
    raise KeyError(f"{inst.task_name}: no gold key in {sorted(res)}")


def _call_generate(inst, task: str, size):
    if task in NO_LIST_SIZE:
        return inst.generate_data()
    return inst.generate_data(size)


def build_items() -> list[dict]:
    """The 8,400 items of the protocol, generated with the official code. Row: id, task, group, list_size, index, min_val, max_val, prompt, data_point (JSON text), gold (JSON text)."""
    rows = []
    for group, task, size, lo, hi in configs():
        inst = make_task(task, lo, hi)
        data = _call_generate(inst, task, size)
        if len(data) != DATAPOINTS:
            raise RuntimeError(f"{task} size {size}: {len(data)} items, expected {DATAPOINTS}")
        for i, dp in enumerate(data):
            dp_json = json.dumps(dp, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            if json.loads(dp_json) != dp:
                raise RuntimeError(f"{task}: data point does not survive JSON: {dp!r}")
            gold = gold_of(inst, dp)
            rows.append({
                "id": f"{task}|{size if size is not None else '-'}|{i}", "task": task, "group": group, "list_size": size, "index": i, "min_val": lo, "max_val": hi,
                "prompt": inst.create_prompt(dp), "data_point": dp_json,
                "gold": json.dumps(gold, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
            })
    if len(rows) != N_ITEMS:
        raise RuntimeError(f"{len(rows)} items, expected {N_ITEMS}")
    return rows


def dumps_rows(rows: list[dict]) -> bytes:
    """The canonical bytes of the prepared file (the sha256 pinned in ITEMS_SHA256 is of these)."""
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")


def items_dir() -> Path:
    """<cache>/xdg/beyondbench_easy_v021: $TITAN_EVAL_CACHE, else the parent of $HF_HOME (as the other prepared files, agieval.agieval_dir)."""
    import os
    root = os.environ.get("TITAN_EVAL_CACHE") or (str(Path(os.environ["HF_HOME"]).parent) if os.environ.get("HF_HOME") else "")
    return Path(root or ".") / "xdg" / ITEMS_FOLDER


def load_items(task: str | None = None, path: Path | None = None) -> list[dict]:
    """The prepared rows (all, or one task's), after the checks: the file's sha256 is the pinned one and it holds N_ITEMS rows."""
    p = path or items_dir() / ITEMS_FILE
    if not p.is_file():
        raise FileNotFoundError(f"{p}: the BeyondBench items are not prepared; on the login node: titan-eval prepare beyondbench")
    raw = p.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != ITEMS_SHA256:
        raise ValueError(f"{p}: sha256 {got} is not the pinned {ITEMS_SHA256} (BeyondBench {OFFICIAL_VERSION}, seed {SEED})")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != N_ITEMS:
        raise ValueError(f"{p}: {len(rows)} rows, the pinned file has {N_ITEMS}")
    return rows if task is None else [r for r in rows if r["task"] == task]


# ====================================================================================================================== scoring

class _Expired(BaseException):
    """The deadline. A BaseException on purpose (scripts/beyondbench_eval.py): some official parsers catch `Exception` inside, a deadline must escape them."""


class _Deadline:
    """SIGALRM deadline of the owner's wrapper (setitimer). Needs the main thread of the process (lighteval computes its metrics there); in another thread there is no deadline
    (`armed` False) and the caller is told."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.armed = False

    def __enter__(self):
        if self.seconds and threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer"):
            def expired(signum, frame):
                raise _Expired()
            self._prev = signal.signal(signal.SIGALRM, expired)
            self._prev_timer = signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self._t0 = time.monotonic()
            self.armed = True
        return self

    def __exit__(self, *exc):
        if self.armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._prev)
            if self._prev_timer[0] > 0:                       # an outer timer was running: give it back what is left
                signal.setitimer(signal.ITIMER_REAL, max(1e-6, self._prev_timer[0] - (time.monotonic() - self._t0)), self._prev_timer[1])
        return False


_INSTANCES: dict[tuple[str, bool], object] = {}
FIXED_TASKS = ("second_minimum",)                           # tasks whose titan_eval verdict (`fixed=True`) differs from the official one


def _second_minimum_fixed_class() -> type:
    """The official SecondMinimumTask with ONE change: the answer is read with the boxed-answer parser `parse_arithmetic_result` (= parse_boxed_answer(response, "number"), the parser of
    find_maximum, find_minimum, mean, median, ...) instead of `parse_number(response)`, which takes the first number of the whole reply. The gold, the rounding and the comparison are the official lines."""
    base = task_class("second_minimum")
    parsing = importlib.import_module(f"{OFFICIAL_PACKAGE}.utils.parsing")

    class SecondMinimumBoxed(base):
        def evaluate_response(self, response, data_point):
            distinct = sorted(set(data_point))
            ground_truth = distinct[1]
            parsed = parsing.parse_arithmetic_result(response)
            instruction_followed = parsed is not None
            accuracy = 0
            if instruction_followed:
                accuracy = 1 if int(round(parsed)) == ground_truth else 0
            return {"input_list": data_point, "ground_truth": ground_truth, "predicted_answer": parsed, "accuracy": accuracy, "instruction_followed": int(instruction_followed)}

    return SecondMinimumBoxed


def _instance(task: str, fixed: bool = False):
    key = (task, fixed and task in FIXED_TASKS)
    if key not in _INSTANCES:
        if key[1]:
            _INSTANCES[key] = _second_minimum_fixed_class()(min_val=-100, max_val=100, num_folds=1, num_samples=DATAPOINTS, seed=SEED)
        else:
            _INSTANCES[key] = make_task(task)                 # evaluate_response reads no range / seed (only the task class's own helpers)
    return _INSTANCES[key]


def _repr(v) -> str | None:
    return None if v is None else repr(v)


def official_verdict(task: str, data_point, response: str, seconds: float = EVAL_TIMEOUT_SECONDS, fixed: bool = False) -> dict:
    """What the package's BaseTask._run_fold_batch makes of one reply: {"accuracy", "instruction_followed", "parsing_success", "generation_success", "timeout", "error", "result"}.
    fixed=True: the documented fix of FIXED_TASKS instead of the official class (everything else is the same code)."""
    out = {"accuracy": 0.0, "instruction_followed": 0, "parsing_success": False, "generation_success": bool(response), "timeout": False, "error": None, "result": None}
    if not response:
        out["error"] = "Empty response"
        return out
    inst = _instance(task, fixed)
    try:
        with _Deadline(seconds):
            res = type(inst).evaluate_response(inst, response, json.loads(json.dumps(data_point)))
    except _Expired:
        out["timeout"] = True
        out["error"] = f"Response evaluation exceeded {seconds:g}s"
        return out
    except Exception as e:                                  # the official loop: accuracy 0, parsing_success False
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out.update(accuracy=float(res.get("accuracy", 0) or 0), instruction_followed=int(bool(res.get("instruction_followed"))), parsing_success=True, result=res)
    return out


def parsed_of(result: dict | None):
    if not result:
        return None
    for k in _PARSED_KEYS:
        if k in result:
            return result[k]
    return None


def score_item(task: str, data_point, response: str, seconds: float = EVAL_TIMEOUT_SECONDS) -> dict:
    """Per item: `correct` (titan_eval's verdict: the official one, with the fix of FIXED_TASKS), `official_correct` (the package's own verdict, bug included), empty, timeout, eval_error
    (an exception inside the scorer other than the deadline), instruction_followed, parsed (repr of the parsed answer, for the details), error."""
    off = official_verdict(task, data_point, response, seconds)
    v = official_verdict(task, data_point, response, seconds, fixed=True) if task in FIXED_TASKS else off
    empty = not response
    timeout = bool(v["timeout"])
    err = (not empty) and (not timeout) and (not v["parsing_success"])
    return {"correct": v["accuracy"], "official_correct": off["accuracy"], "empty": float(empty), "timeout": float(timeout), "eval_error": float(err),
            "instruction_followed": float(v["instruction_followed"]), "official_instruction_followed": float(off["instruction_followed"]), "parsed": _repr(parsed_of(v["result"])),
            "error": v["error"]}


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
