# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
name:
ZebraLogic (chat, titan_eval): zebralogic_chat

dataset:
allenai/ZebraLogicBench-private, config grid_mode, split test (1,000 puzzles; gated, the official solutions), rev 9f39ef49

abstract:
Official ZeroEval grid-mode protocol (WildEval/ZeroEval, src/templates/ZEBRA_GRID.py + src/evaluation/zebra_grid_eval.py) for models used through
their chat template: the prompt is the ZEBRA_GRID template with the puzzle and the JSON answer template (every cell "___"), 0-shot, the
answer is the JSON after the LAST </think> (an unclosed block gives no answer). Scoring: a cell is correct when its lowercased, stripped value
equals the official solution's; puzzle accuracy = all cells correct (primary); cell accuracy = correct cells / cells per puzzle, averaged over
puzzles (a documented deviation: the official cell accuracy is pooled over all cells, cell-weighted); size buckets small / medium / large / xl as
in the official script (their sizes are the official buckets; an unparsed answer scores 0).
Single sample, token budget BUDGET (32,768; the run's gen.max_new_tokens replaces it).

languages:
english

tags:
chat, reasoning, logic

paper:
https://github.com/WildEval/ZeroEval (ZebraLogic grid mode, the official protocol); https://huggingface.co/datasets/allenai/ZebraLogicBench-private
"""

import json
import re

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.chat_evals import BUDGET


ZEBRA_REVISION = "9f39ef490ae924437376657205025f26c0bd1af3"

# WildEval/ZeroEval src/templates/ZEBRA_GRID.py, verbatim (the template text is the prompt; {puzzle} and {json_template} are filled in)
ZEBRA_GRID = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {
        "House 1": {
            "Name": "Arnold",
            "Drink": "tea"
        },
        "House 2": {
            "Name": "Peter",
            "Drink": "water"
        },
        "House 3": {
            "Name": "Eric",
            "Drink": "milk"
        }
    }
}

# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""

# ZeroEval's size buckets (zebra_grid_eval.py); every size of the dataset is in exactly one of them
SIZE_BUCKETS = {
    "small": ["2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3", "4*2"],
    "medium": ["3*4", "3*5", "3*6", "4*3", "4*4", "5*2", "6*2"],
    "large": ["4*5", "5*3", "4*6", "5*4", "6*3"],
    "xl": ["5*5", "6*4", "5*6", "6*5", "6*6"],
}
BUCKET_OF = {size: bucket for bucket, sizes in SIZE_BUCKETS.items() for size in sizes}
BUCKET_KEYS = [f"{b}_acc" for b in SIZE_BUCKETS]


def json_template(solution: dict) -> str:
    """The answer template of the prompt: the house count and the attribute columns of the puzzle, every value '___' (as the official run)."""
    header = solution["header"]
    template = {"reasoning": "___", "solution": {}}
    for i in range(len(solution["rows"])):
        template["solution"][f"House {i + 1}"] = {header[j]: "___" for j in range(1, len(header))}
    return json.dumps(template, indent=4)


def zebra_chat_prompt(line, task_name: str = None):
    solution = line["solution"]
    query = ZEBRA_GRID.replace("{puzzle}", line["puzzle"]).replace("{json_template}", json_template(solution))
    table = {}
    header = solution["header"]
    for i, row in enumerate(solution["rows"]):
        table[f"House {i + 1}"] = {header[j]: row[j] for j in range(1, len(header))}
    return Doc(
        task_name=task_name,
        query=query,                    # no `instruction`: the prompt is the official template, the prompt manager adds nothing
        choices=[""],
        gold_index=0,
        specific={"solution_table": table, "size": line["size"], "id": line["id"]},
    )


def extract_last_complete_json(text: str):
    """ZeroEval eval_utils.extract_last_complete_json: the last top-level {...} object of the text (braces matched with a stack), parsed
    after removing its newlines; None when there is none or it does not parse."""
    stack = []
    last_start = None
    last_str = None
    for i, char in enumerate(text):
        if char == "{":
            stack.append(i)
            if last_start is None:
                last_start = i
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    last_str = text[last_start : i + 1]
                    last_start = None
    if last_str:
        try:
            return json.loads(last_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    return None


def _cell_value(value):
    """The official comparison: a string, or the first item of a list; anything else is a wrong cell."""
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, str):
        return value.lower().strip()
    return None


def score_zebra(solution_table: dict, text: str) -> dict:
    """Official scoring of one answer: puzzle (all cells right), cell counts, and whether a usable solution was parsed."""
    total = sum(len(cells) for cells in solution_table.values())
    parsed = extract_last_complete_json(text or "")
    pred = parsed.get("solution") if isinstance(parsed, dict) else None
    if not isinstance(pred, dict):
        return {"parsed": 0.0, "correct": 0, "total": total}
    correct = 0
    for house, cells in solution_table.items():
        pred_house = pred.get(house)
        if not isinstance(pred_house, dict):
            continue
        for column, gold in cells.items():
            if column not in pred_house:
                continue
            got = _cell_value(pred_house[column])
            if got is not None and got == str(gold).lower().strip():
                correct += 1
    return {"parsed": 1.0, "correct": correct, "total": total}


class ZebraChatMetrics(SampleLevelComputation):
    """puzzle_acc (primary), cell_acc, parsed, and the size bucket of the puzzle (its own key; None for the other buckets)."""

    def compute(self, doc: Doc, model_response, **kwargs) -> dict:
        text = model_response.final_text[0] if model_response.final_text else ""
        s = score_zebra(doc.specific["solution_table"], text)
        puzzle = float(s["correct"] == s["total"])
        out = {
            "puzzle_acc": puzzle,
            "cell_acc": s["correct"] / s["total"],
            "parsed": s["parsed"],
        }
        bucket = BUCKET_OF.get(doc.specific["size"])
        for b in SIZE_BUCKETS:
            out[f"{b}_acc"] = puzzle if bucket == b else None
        return out


def _mean(values) -> float:
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else 0.0


ZEBRA_METRICS = SampleLevelMetricGrouping(
    metric_name=["puzzle_acc", "cell_acc", "parsed", *BUCKET_KEYS],
    sample_level_fn=ZebraChatMetrics(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={name: _mean for name in ["puzzle_acc", "cell_acc", "parsed", *BUCKET_KEYS]},
    higher_is_better={name: True for name in ["puzzle_acc", "cell_acc", "parsed", *BUCKET_KEYS]},
)


TASKS_TABLE = [
    LightevalTaskConfig(
        name="zebralogic_chat",
        prompt_function=zebra_chat_prompt,
        hf_repo="allenai/ZebraLogicBench-private",
        hf_subset="grid_mode",
        hf_revision=ZEBRA_REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[ZEBRA_METRICS],
        version=0,
    ),
]
