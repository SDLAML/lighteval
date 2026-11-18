"""
name:
BeyondBench (easy suite, base)

dataset:
BeyondBench 0.2.1 easy suite, generated with the official generators (seed 42, 100 items per task and list size; prepared file under <cache>/xdg/beyondbench_easy_v021)

abstract:
BeyondBench (arXiv 2509.24210, ICLR 2026) generates reasoning problems on the fly so that a model cannot have seen them. The easy suite has 44 list / arithmetic tasks
(sum, sorting, median, count, set operations, ...). The items here are the owner's protocol of scripts/run_beyondbench.sh: 84 (task, list size) configurations x 100 = 8,400 items,
greedy, 4,096 new tokens, no chat template, the official scorer. Module beyondbench_items.py has the generator and the scorer.

languages:
english

tags:
math, reasoning, synthetic

paper:
https://arxiv.org/abs/2509.24210
"""

import json

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks import beyondbench_items as bb


BEYONDBENCH_REPO = "beyondbench_local"          # the `hf_repo` of the tasks; `load_datasets` (lighteval_task.py) reads the subset `hf_subset` (= the task) with `load_beyondbench`
BEYONDBENCH_REVISION = bb.OFFICIAL_VERSION       # "0.2.1": the official package version the items were generated with (the lock row ctrl-gaurav/BeyondBench@0.2.1, kind files)
NAN = float("nan")


def load_beyondbench(subset: str):
    """The rows of one task as a DatasetDict with ONE split, `train`. The file's sha256 is checked against the pin (bb.ITEMS_SHA256) first."""
    from datasets import Dataset, DatasetDict

    if subset not in bb.TASKS:
        raise KeyError(f"{subset!r} is not one of the {len(bb.TASKS)} easy tasks")
    return DatasetDict({"train": Dataset.from_list(bb.load_items(subset))})


def beyondbench_prompt(line, task_name: str = None) -> Doc:
    """The official prompt, unchanged, as the whole query (0-shot, no instruction line, no answer prefix: the base prompt of the owner's `--no-chat-template` run). The gold (JSON text) is the
    only choice; the scorer does not read it (the official evaluate_response recomputes the gold from the data point, which `specific` carries)."""
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[line["gold"]],
        gold_index=0,
        specific={"item": line["id"], "task": line["task"], "group": line["group"], "list_size": line["list_size"], "index": line["index"], "data_point": json.loads(line["data_point"])},
    )


NAMES = ("acc", "acc_official", "instruction_followed", "empty", "timeout", "eval_error")


class _BeyondBenchScore(SampleLevelComputation):
    """Per item: `acc` (titan_eval's verdict = the official one except the documented `second_minimum` fix; an empty / failed / timed-out reply is 0 and stays in the denominator), `acc_official`
    (the package's own verdict and denominator: NaN for an empty reply = left out, as the official BaseTask.process_fold_metrics does; `second_minimum` with its bug), `instruction_followed` (the parsers found an answer), `empty` / `timeout` / `eval_error` (0 / 1 flags). The parsed answer and the gold go to the
    details through `doc.specific` (`extracted_predictions` / `extracted_golds`, the keys the math metrics use)."""

    def compute(self, doc, model_response, **kwargs) -> dict[str, float]:
        spec = doc.specific
        texts = list(model_response.final_text or [])
        text = texts[0] if texts else ""
        s = bb.score_item(spec["task"], spec["data_point"], text)
        spec["extracted_predictions"] = [] if s["parsed"] is None else [s["parsed"]]
        spec["extracted_golds"] = [doc.choices[0]]
        spec["verdict"] = "empty" if s["empty"] else "timeout" if s["timeout"] else "eval_error" if s["eval_error"] else ("correct" if s["correct"] else "wrong")
        return {
            "acc": s["correct"],
            "acc_official": NAN if s["empty"] else s["official_correct"],
            "instruction_followed": s["instruction_followed"],
            "empty": s["empty"],
            "timeout": s["timeout"],
            "eval_error": s["eval_error"],
        }


def _avg_nonempty(values) -> float:
    """Average over the items that are not NaN (an empty reply is NaN in `acc_official`). Named without "mean" on purpose (as chat_long._members_avg): lighteval gives a "mean" aggregation the plain
    stderr, which the NaN would poison; any other name is bootstrapped over the non-NaN-aware function."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


beyondbench_metric = SampleLevelMetricGrouping(
    metric_name=list(NAMES),
    sample_level_fn=_BeyondBenchScore(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={n: (_avg_nonempty if n == "acc_official" else np.mean) for n in NAMES},
    higher_is_better={"acc": True, "acc_official": True, "instruction_followed": True, "empty": False, "timeout": False, "eval_error": False},
)


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"beyondbench_easy_base:{task}",
        prompt_function=beyondbench_prompt,
        hf_repo=BEYONDBENCH_REPO,
        hf_revision=BEYONDBENCH_REVISION,
        hf_subset=task,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=bb.MAX_NEW_TOKENS,          # the owner's --max-tokens 4096; greedy comes from the run's gen defaults (temperature 0, top_p 1)
        metrics=[beyondbench_metric],
        stop_sequence=[],                           # none: the official run has none (the model ends with its EOS or at the cap)
        version=0,
    )
    for task in bb.TASKS
]
