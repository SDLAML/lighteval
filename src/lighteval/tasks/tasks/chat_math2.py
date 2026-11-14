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
Chat evals, second set (titan_eval, post-trained models): IMO-AnswerBench, OMEGA, MMLU-ProX (28 languages, chat prompt)

dataset:
google-deepmind/superhuman imobench/answerbench_v2.csv (commit 955558ad, CC-BY-4.0 / Apache-2.0 for code; read from the cache, see IMO_DIR),
allenai/omega-explorative / omega-compositional / omega-transformative (MIT), li-lab/MMLU-ProX @ 8e6106a6 (MIT)

abstract:
Same conventions as chat_evals.py (0-shot, the model's chat template, the scored text = what follows the LAST </think>, metrics in
lighteval/metrics/chat_metrics.py). Subsets (stored, chat_math2_subsets.json, drawn by `python -m lighteval.tasks.tasks.chat_math2 --draw`):
  - IMO-AnswerBench: all 400 problems of answerbench_v2.csv (the v1 file is deprecated upstream; the HF mirrors are v1). Math prompt of math_500_chat.
    Gold = the 'Short Answer' column, wrapped as \\boxed{...}. The official grader is an LLM (AnswerAutoGrader, Gemini 2.5 Pro); this
    task is rule-checked (math-verify style), so answers the rule cannot compare are counted, not guessed (see the review page).
  - OMEGA: per (broad category, subset, split) bucket 20 seeded items of the test splits (explorative: test_in + test_out for all 41 subsets;
    compositional and transformative: test). The OLMES prompt of OMEGA (olmo3:adapt): question + "Therefore, the final answer is \\boxed{...}".
    Subtask names omega_chat:<category>:<subset>:<split>.
  - MMLU-ProX: 28 languages (the base task mmlu_prox:mcf_em languages), test split, 200 seeded items per language (question_id), the
    MMLU-Pro chat prompt (mc_query of chat_evals: English instruction, "Answer: $LETTER", the letters of the question, 3 to 10 options).
    Subtask names mmlu_prox_chat:<lang>.
"""

import hashlib
import json
import os
import random
from functools import partial
from pathlib import Path
from string import ascii_uppercase

from lighteval.metrics.chat_metrics import chat_choice_metric, chat_math_metric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.chat_evals import BUDGET, MATH_PROMPT, mc_query


SEED = "chat_math2-v1"
SUBSETS_FILE = Path(__file__).with_name("chat_math2_subsets.json")

# ----------------------------------------------------------------------------------------------------- IMO-AnswerBench
IMO_FILE = "answerbench_v2.csv"
IMO_REVISION = "955558ad67ef24dccd27545c326988bdad1b4471"     # last commit of imobench/answerbench_v2.csv (2026-02-13)
IMO_SHA256 = "275877a9d988d85278fad3a5f8a41d7f83393a60bf259531ec0a5161e6b21cf9"


def imo_dir() -> str:
    """The folder of the pinned CSV: <cache>/xdg/imo_bench (datasets_lock add-file, root xdg). The cache is $TITAN_EVAL_CACHE (set by serve/env.sh),
    else the parent of $HF_HOME. Empty when neither is set: the load then fails with the path in the error."""
    root = os.environ.get("TITAN_EVAL_CACHE") or (str(Path(os.environ["HF_HOME"]).parent) if os.environ.get("HF_HOME") else "")
    return str(Path(root) / "xdg" / "imo_bench") if root else ""


def _boxed(answer: str) -> str:
    """The gold text wrapped as a boxed answer for the extractor (outer $ of the dataset text removed; the extractor reads the box first)."""
    a = str(answer).strip()
    if a.startswith("$") and a.endswith("$") and len(a) > 1:
        a = a[1:-1].strip()
    return "\\boxed{" + a + "}"


def imo_chat_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT.format(problem=line["Problem"].strip()),
        choices=[_boxed(line["Short Answer"])],
        gold_index=0,
        specific={"problem_id": line["Problem ID"], "category": line["Category"], "subcategory": line["Subcategory"]},
    )


# --------------------------------------------------------------------------------------------------------------- OMEGA
OMEGA_REPOS = {
    "explorative": ("allenai/omega-explorative", "b04ae8d4757a2229e8ed65ac6a923e502d0bbb95", ["test_in", "test_out"]),
    "compositional": ("allenai/omega-compositional", "fabf91c0e85ab3ba5ec2ecfc38a2f30a6d0fadb9", ["test"]),
    "transformative": ("allenai/omega-transformative", "f188d999ef40f51843828d6d268c720feb3d355a", ["test"]),
}
OMEGA_PER_BUCKET = 20

# OLMES omega (olmo3:adapt): the problem with the \boxed instruction removed, then this template.
OMEGA_TEMPLATE = (
    '{question}\n\nShow your work and conclude with "Therefore, the final answer is \\boxed{{answer}}." where answer is just the final solution '
    'that solves the problem. E.g., if the answer is 6, conclude with "Therefore, the final answer is \\boxed{{6}}."'
)


def _omega_question(messages) -> str:
    text = messages[0]["content"]
    return text.replace("\n\nPresent the answer in LaTex format: \\boxed{Your answer}", "")


def omega_chat_prompt(line, task_name: str = None):
    question = _omega_question(line["messages"])
    return Doc(
        task_name=task_name,
        query=OMEGA_TEMPLATE.format(question=question),
        choices=[_boxed(line["ground_truth"])],
        gold_index=0,
        specific={"id": line["id"], "setting_key": line.get("setting_key")},
    )


def omega_filter(subset_ids: frozenset):
    def keep(row) -> bool:
        return row["id"] in subset_ids

    return keep


# ------------------------------------------------------------------------------------------------------------ MMLU-ProX
MMLU_PROX_REPO = "li-lab/MMLU-ProX"
MMLU_PROX_REVISION = "8e6106a6c6ce1c5027e66cc338143cf997b2aa09"
MMLU_PROX_LANGS = [
    "af", "ar", "bn", "cs", "de", "es", "fr", "hi", "hu", "id", "it", "ja", "ko", "mr", "ne", "pt", "ru", "sr", "sw", "te", "th", "uk", "ur",
    "vi", "wo", "yo", "zh", "zu",
]                                                              # the languages of the base task mmlu_prox:mcf_em (English is mmlu_pro_chat)
MMLU_PROX_PER_LANG = 200


def _options(line):
    return [str(line[f"option_{i}"]) for i in range(10) if line.get(f"option_{i}") is not None and str(line[f"option_{i}"]).strip()]


def mmlu_prox_chat_prompt(line, task_name: str = None):
    options = _options(line)
    return Doc(
        task_name=task_name,
        query=mc_query(line["question"], options),
        choices=list(ascii_uppercase[: len(options)]),
        gold_index=int(line["answer_index"]),
        specific={"question_id": line["question_id"], "category": line.get("category"), "src": line.get("src")},
    )


def _load_subsets() -> dict:
    if not SUBSETS_FILE.exists():
        raise FileNotFoundError(f"{SUBSETS_FILE} is missing: run python -m lighteval.tasks.tasks.chat_math2 --draw (login node, datasets cached)")
    return json.loads(SUBSETS_FILE.read_text())


def mmlu_prox_filter(lang: str, ids: frozenset):
    def keep(row) -> bool:
        return row["question_id"] in ids

    return keep


# ------------------------------------------------------------------------------------------------------------- tasks
def _build_tasks():
    tasks = []
    # IMO-AnswerBench: local CSV (no hub repo). The folder's csv is checked against the pinned sha256 before it is used (a swapped file is an error).
    csv_path = Path(imo_dir() or ".") / IMO_FILE
    if csv_path.exists():
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if digest != IMO_SHA256:
            raise ValueError(f"{csv_path}: sha256 {digest} is not the pinned {IMO_SHA256} (commit {IMO_REVISION})")
    tasks.append(
        LightevalTaskConfig(
            name="imo_answerbench_chat",
            prompt_function=imo_chat_prompt,
            hf_repo=imo_dir(),
            hf_subset="default",
            hf_revision=None,
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=BUDGET,
            stop_sequence=[],
            metrics=[chat_math_metric(1)],
            version=0,
        )
    )
    if not SUBSETS_FILE.exists():
        return tasks
    subsets = _load_subsets()
    for category, (repo, revision, splits) in OMEGA_REPOS.items():
        for key, ids in sorted(subsets["omega"].items()):
            cat, subset, split = key.split("|")
            if cat != category:
                continue
            tasks.append(
                LightevalTaskConfig(
                    name=f"omega_chat:{category}:{subset}:{split}",
                    prompt_function=omega_chat_prompt,
                    hf_repo=repo,
                    hf_subset=subset,
                    hf_revision=revision,
                    hf_avail_splits=[split],
                    evaluation_splits=[split],
                    few_shots_split=None,
                    few_shots_select=None,
                    generation_size=BUDGET,
                    stop_sequence=[],
                    hf_filter=omega_filter(frozenset(ids)),
                    metrics=[chat_math_metric(1)],
                    version=0,
                )
            )
    for lang in MMLU_PROX_LANGS:
        ids = frozenset(subsets["mmlu_prox"][lang])
        tasks.append(
            LightevalTaskConfig(
                name=f"mmlu_prox_chat:{lang}",
                prompt_function=mmlu_prox_chat_prompt,
                hf_repo=MMLU_PROX_REPO,
                hf_subset=lang,
                hf_revision=MMLU_PROX_REVISION,
                hf_avail_splits=["test"],
                evaluation_splits=["test"],
                few_shots_split=None,
                few_shots_select=None,
                generation_size=BUDGET,
                stop_sequence=[],
                hf_filter=mmlu_prox_filter(lang, ids),
                metrics=[chat_choice_metric(1)],
                version=0,
            )
        )
    return tasks


TASKS_TABLE = _build_tasks()


# -------------------------------------------------------------------------------------------------------- the draw
def draw(out: Path = SUBSETS_FILE) -> dict:
    """Draw the seeded subsets from the pinned datasets and write them (run once on the login node; the file is the record).
    Seeds: random.Random(f"{SEED}:{bucket}") over the SORTED ids of the bucket, so the draw is the same on every machine."""
    from datasets import load_dataset

    omega = {}
    for category, (repo, revision, splits) in OMEGA_REPOS.items():
        configs = sorted({c for c in _omega_configs(repo, revision)})
        for subset in configs:
            for split in splits:
                ds = load_dataset(repo, subset, split=split, revision=revision)
                ids = sorted(ds["id"])
                k = min(OMEGA_PER_BUCKET, len(ids))
                pick = random.Random(f"{SEED}:omega:{category}:{subset}:{split}").sample(ids, k)
                omega[f"{category}|{subset}|{split}"] = sorted(pick)
    mmlu = {}
    for lang in MMLU_PROX_LANGS:
        ds = load_dataset(MMLU_PROX_REPO, lang, split="test", revision=MMLU_PROX_REVISION)
        qids = sorted(int(q) for q in ds["question_id"])
        mmlu[lang] = sorted(random.Random(f"{SEED}:mmlu_prox:{lang}").sample(qids, MMLU_PROX_PER_LANG))
    data = {"seed": SEED, "omega": omega, "mmlu_prox": mmlu,
            "omega_per_bucket": OMEGA_PER_BUCKET, "mmlu_prox_per_language": MMLU_PROX_PER_LANG}
    out.write_text(json.dumps(data, indent=0, sort_keys=True) + "\n")
    return data


def _omega_configs(repo: str, revision: str) -> list[str]:
    """The config (subset) folders of an OMEGA repo at the pinned revision (one folder per config, parquet inside)."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, revision=revision)
    names = []
    for sib in info.siblings:
        parts = sib.rfilename.split("/")
        if len(parts) == 2 and parts[0] not in names and not parts[0].startswith("."):
            names.append(parts[0])
    return names


if __name__ == "__main__":
    import sys

    if "--draw" in sys.argv:
        d = draw()
        print(f"omega buckets {len(d['omega'])}, mmlu_prox languages {len(d['mmlu_prox'])}")
    else:
        print("usage: python -m lighteval.tasks.tasks.chat_math2 --draw")
