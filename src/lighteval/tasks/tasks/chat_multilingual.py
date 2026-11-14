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
Chat evals, multilingual set (titan_eval, post-trained models): MMMLU (14 languages), INCLUDE (44 languages)

dataset:
openai/MMMLU @ 325a01dc (per-locale configs, test split; MIT, the card's license),
CohereLabs/include-base-44 @ d2e1f601 (the fork's base revision; Apache-2.0)

abstract:
Same conventions as chat_evals.py (0-shot, the model's chat template, the scored text = what follows the LAST </think>, metrics in
lighteval/metrics/chat_metrics.py). The question and options are in their own language; the instruction is the English MMLU-Pro chat
prompt of chat_evals.mc_query ("Answer: $LETTER"), the same choice as mmlu_prox_chat (chat_math2.py). Subsets (stored in
chat_multilingual_subsets.json, drawn by `python -m lighteval.tasks.tasks.chat_multilingual --draw`, seed chat_multilingual-v1):
  - MMMLU: 14 locales (the report's 14 languages: AR_XY, BN_BD, DE_DE, ES_LA, FR_FR, HI_IN, ID_ID, IT_IT, JA_JP, KO_KR, PT_BR, SW_KE,
    YO_NG, ZH_CN), 200 seeded items per locale (the locale has 14,042 test rows; key = the row's `Unnamed: 0` index). Subtask names
    mmmlu_chat:<lang> with <lang> = ar, bn, de, es, fr, hi, id, it, ja, ko, pt, sw, yo, zh.
  - INCLUDE: 44 languages (include-base-44 test split, 22,639 questions; the fork's include.py language map), 200 seeded items per language
    (key = md5 of question + options + answer, the base file has no id column). Languages with fewer than 200 questions keep all of them.
    Subtask names include_chat:<iso3>; the registry's mean over subtasks is the mean over languages.
The report sampled 10% of MMMLU and INCLUDE (Qwen3 report, see the review page); this task uses a fixed 200 per language instead.
"""

import hashlib
import json
import random
from pathlib import Path
from string import ascii_uppercase

from lighteval.metrics.chat_metrics import chat_choice_metric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks.include import _BASE_REPO, _BASE_REVISION, _LANGUAGES, _choices
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.chat_evals import BUDGET, mc_query


SEED = "chat_multilingual-v1"
SUBSETS_FILE = Path(__file__).with_name("chat_multilingual_subsets.json")
PER_LANG = 200

# ----------------------------------------------------------------------------------------------------------------- MMMLU
MMMLU_REPO = "openai/MMMLU"
MMMLU_REVISION = "325a01dc3e173cac1578df94120499aaca2e2504"   # main on 2026-11-14; per-locale configs (the base task's 038c7808 holds CSV files only)
# locale config -> lighteval subtask code (ISO 639-1; the 14 languages of the Qwen3 report's MMMLU row)
MMMLU_LOCALES = {
    "AR_XY": "ar", "BN_BD": "bn", "DE_DE": "de", "ES_LA": "es", "FR_FR": "fr", "HI_IN": "hi", "ID_ID": "id",
    "IT_IT": "it", "JA_JP": "ja", "KO_KR": "ko", "PT_BR": "pt", "SW_KE": "sw", "YO_NG": "yo", "ZH_CN": "zh",
}


def mmmlu_chat_prompt(line, task_name: str = None):
    options = [str(line[k]) for k in ("A", "B", "C", "D")]
    return Doc(
        task_name=task_name,
        query=mc_query(line["Question"], options),
        choices=list(ascii_uppercase[:4]),
        gold_index=ascii_uppercase.index(line["Answer"]),
        specific={"subject": line.get("Subject"), "row": line.get("Unnamed: 0")},
    )


def mmmlu_filter(ids: frozenset):
    def keep(row) -> bool:
        return int(row["Unnamed: 0"]) in ids

    return keep


# -------------------------------------------------------------------------------------------------------------- INCLUDE
def include_row_key(line) -> str:
    """Stable id of an include-base-44 row (the base file has no id): md5 of question, the four options and the answer."""
    parts = [line["question"], line["option_a"], line["option_b"], line["option_c"], line["option_d"], line["answer"]]
    return hashlib.md5("\x1f".join(str(p) for p in parts).encode()).hexdigest()


def include_chat_prompt(line, task_name: str = None):
    options = _choices(line)
    return Doc(
        task_name=task_name,
        query=mc_query(line["question"], options),
        choices=list(ascii_uppercase[: len(options)]),
        gold_index=int(line["answer"]),
        specific={"subject": line.get("subject"), "domain": line.get("domain"), "country": line.get("country"),
                  "level": line.get("level"), "regional_feature": line.get("regional_feature")},
    )


def include_filter(ids: frozenset):
    def keep(row) -> bool:
        return include_row_key(row) in ids

    return keep


# ------------------------------------------------------------------------------------------------------------- tasks
def _load_subsets() -> dict:
    return json.loads(SUBSETS_FILE.read_text())


def _build_tasks():
    tasks = []
    if not SUBSETS_FILE.exists():
        return tasks
    subsets = _load_subsets()
    for locale, code in MMMLU_LOCALES.items():
        ids = frozenset(subsets["mmmlu"][locale])
        tasks.append(
            LightevalTaskConfig(
                name=f"mmmlu_chat:{code}",
                prompt_function=mmmlu_chat_prompt,
                hf_repo=MMMLU_REPO,
                hf_subset=locale,
                hf_revision=MMMLU_REVISION,
                hf_avail_splits=["test"],
                evaluation_splits=["test"],
                few_shots_split=None,
                few_shots_select=None,
                generation_size=BUDGET,
                stop_sequence=[],
                hf_filter=mmmlu_filter(ids),
                metrics=[chat_choice_metric(1)],
                version=0,
            )
        )
    for language, code in _LANGUAGES.items():
        ids = frozenset(subsets["include"][language])
        tasks.append(
            LightevalTaskConfig(
                name=f"include_chat:{code}",
                prompt_function=include_chat_prompt,
                hf_repo=_BASE_REPO,
                hf_subset=language,
                hf_revision=_BASE_REVISION,
                hf_avail_splits=["test"],
                evaluation_splits=["test"],
                few_shots_split=None,
                few_shots_select=None,
                generation_size=BUDGET,
                stop_sequence=[],
                hf_filter=include_filter(ids),
                metrics=[chat_choice_metric(1)],
                version=0,
            )
        )
    return tasks


TASKS_TABLE = _build_tasks()


# -------------------------------------------------------------------------------------------------------- the draw
def draw(out: Path = SUBSETS_FILE) -> dict:
    """Draw the seeded subsets from the pinned datasets and write them (run once on the login node; the file is the record).
    Seeds: random.Random(f"{SEED}:<bucket>") over the SORTED ids of the bucket, so the draw is the same on every machine."""
    from datasets import load_dataset

    mmmlu = {}
    for locale in MMMLU_LOCALES:
        ds = load_dataset(MMMLU_REPO, locale, split="test", revision=MMMLU_REVISION)
        rows = sorted({int(i) for i in ds["Unnamed: 0"]})
        mmmlu[locale] = sorted(random.Random(f"{SEED}:mmmlu:{locale}").sample(rows, min(PER_LANG, len(rows))))
    include = {}
    for language in _LANGUAGES:
        ds = load_dataset(_BASE_REPO, language, split="test", revision=_BASE_REVISION)
        keys = sorted({include_row_key(r) for r in ds})
        include[language] = sorted(random.Random(f"{SEED}:include:{language}").sample(keys, min(PER_LANG, len(keys))))
    data = {"seed": SEED, "per_language": PER_LANG, "mmmlu": mmmlu, "include": include}
    out.write_text(json.dumps(data, indent=0, sort_keys=True) + "\n")
    return data


if __name__ == "__main__":
    import sys

    if "--draw" in sys.argv:
        d = draw()
        print(f"mmmlu locales {len(d['mmmlu'])}, include languages {len(d['include'])}")
    else:
        print("usage: python -m lighteval.tasks.tasks.chat_multilingual --draw")
