"""Few-shot demonstrations of the reading-comprehension tasks that ship ONE split (xquad: validation, multi_wiki_qa: train), by EuroEval's own rule (owner 11-16).

Not a task module (no TASKS_TABLE). Source: EuroEval, `src/euroeval/generation_utils.py` @ b6288706 (2026-11-14), the framework the MultiWikiQA paper (arXiv 2509.04111) evaluates with
("decoder models were evaluated 2-shot"). `extract_few_shot_examples` (lines 1085-1179) and, for the question-answering task group, `_extract_question_answering_examples` (lines 1267-1311).
Quoted:
    1137:    random_seed = 4242 + itr_idx                                  (itr_idx 0: seed 4242)
    1139:    shuffled_train = dataset["train"].shuffle(seed=random_seed)
    1287:    # Locate the maximum number of tokens that constitutes a short example
    1288:    for max_num_tokens in [512, 1024, 2048, 4096, 8192]:
    1289:        train_with_short_examples = shuffled_train.filter(lambda example: len(example["context"]) < max_num_tokens)
    1292:        num_short_examples = len(train_with_short_examples)
    1293:        if num_short_examples >= num_few_shots: break
    1296:        raise InvalidBenchmark("Could not find enough short examples for few-shot learning.")
    1300:    shuffled_train = train_with_short_examples.shuffle(seed=random_seed)
    1302:    while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
    1303:        example = shuffled_train.select(range(1))[0]
    1307:        few_shot_examples.append(example)
    1308:        shuffled_train = shuffled_train.filter(lambda x: x["context"] != example["context"])
    1177:    random.seed(random_seed)
    1178:    random.shuffle(few_shot_examples)
(the unit is CHARACTERS: `len(example["context"])`, though the variable is called max_num_tokens). So: ONE fixed set of `num_few_shots` shots per language (not per scored item), drawn with seed 4242
from the contexts shorter than 512 characters (1024, 2048, 4096, 8192 only if there are fewer short examples than shots), every pick removes all examples with the same context (so the shots have
distinct contexts), and the picked shots are shown in a final seeded shuffle. XQuAD has no official few-shot protocol of its own; the same EuroEval rule is used for it.
`select_shots` is EuroEval's code on plain lists: its `Dataset.shuffle(seed=s)` is `np.random.default_rng(s).permutation(len(dataset))` (datasets' own implementation), `filter` keeps the order, `select(range(1))[0]` is the
first row; so the picks are what EuroEval's code makes of the same rows (the unit test runs a verbatim copy of EuroEval's code on `datasets.Dataset` and compares). No `datasets` map / filter / shuffle is run on the
cached dataset (they write cache-*.arrow files into the shared dataset cache).
Differences, all forced by the data: (1) EuroEval draws from its train split and scores its test split; our datasets have ONE split, so the shots AND every item that shares a shot's context are held out of
scoring (the base_extra convention of heldout_shots.py; `heldout_contexts`), at every K, so that the scored items are the same for every shot count; (2) a row without any non-empty answer
(multi_wiki_qa: about 1 per language) is left out of the candidate pool (the task cannot build a prompt for it); (3) the local `random.Random(seed)` instead of the global `random.seed`
(the same sequence). K shots = the first K of the set drawn for the task's maximum K (`HELDOUT_SHOTS` of the task module): `|K` with K < max shows a prefix.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Callable

import numpy as np

SEED = 4242                       # EuroEval generation_utils.py line 1137: 4242 + itr_idx, itr_idx = 0
THRESHOLDS = (512, 1024, 2048, 4096, 8192)      # line 1288, characters of the context


def _shuffle(rows: list[dict], seed: int) -> list[dict]:
    """`datasets.Dataset.shuffle(seed=seed)`: the rows in the order of `np.random.default_rng(seed).permutation(len(rows))`."""
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))]


def select_shots(rows, num_shots: int, seed: int = SEED, valid: Callable[[dict], bool] | None = None) -> list[dict]:
    """EuroEval's `_extract_question_answering_examples` + the final shuffle of `extract_few_shot_examples` on the rows (a `datasets.Dataset` or a list of dicts with `context`, `question`, `answers`, ...)."""
    rows = [dict(r) for r in rows]
    if valid is not None:
        rows = [r for r in rows if valid(r)]
    shuffled = _shuffle(rows, seed)
    for limit in THRESHOLDS:
        short = [r for r in shuffled if len(r["context"]) < limit]
        if len(short) >= num_shots:
            break
    else:
        raise ValueError("Could not find enough short examples for few-shot learning.")
    shuffled = _shuffle(short, seed)
    shots: list[dict] = []
    while len(shots) < num_shots and len(shuffled) > 0:
        example = shuffled[0]
        shots.append(example)
        shuffled = [x for x in shuffled if x["context"] != example["context"]]
    random.Random(seed).shuffle(shots)
    return shots


@lru_cache(maxsize=None)
def _load(repo: str, subset: str, revision: str, split: str):
    from lighteval.tasks.lighteval_task import load_dataset_pinned

    return load_dataset_pinned(repo, subset, revision=revision)[split]


def has_answer(row: dict) -> bool:
    return any(len(a) > 0 for a in row["answers"]["text"])


@lru_cache(maxsize=None)
def heldout_contexts(repo: str, subset: str, revision: str, split: str, num_shots: int) -> tuple[tuple[dict, ...], frozenset]:
    """(the shots in the order they are shown, the set of contexts that are not scored): the shots' contexts, i.e. the shots and every other question on the same context."""
    shots = select_shots(_load(repo, subset, revision, split), num_shots, valid=has_answer)
    return tuple(dict(s) for s in shots), frozenset(s["context"] for s in shots)


def scored_only(prompt_function: Callable, held: Callable[[], frozenset]) -> Callable:
    """The prompt function, returning None for an item on a held-out context when it is NOT building a demonstration (`line["__few_shots"]`)."""
    keys = None

    def wrapped(line, task_name: str = None):
        nonlocal keys
        if not line.get("__few_shots"):
            keys = keys if keys is not None else held()
            if line["context"] in keys:
                return None
        return prompt_function(line, task_name)

    return wrapped
