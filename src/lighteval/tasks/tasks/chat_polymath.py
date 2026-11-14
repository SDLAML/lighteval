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
Chat evals, PolyMath and MGSM (titan_eval, post-trained models): polymath_chat (18 languages x 4 difficulty levels), mgsm_chat (40 languages)

dataset:
Qwen/PolyMath @ 71e0db90 (per-language configs, splits low / medium / high / top, 125 problems per language and level; the card names
no license, arXiv 2504.18428, code QwenLM/PolyMath @ fbf4e41c: instruction.py for the answer-format note, eval/run_eval.py for the
level weights and the official scorer),
CohereLabs/global-mgsm @ 29087245 (the base mgsm:gen revision; test split, 250 problems per language, no English; the same pin).

abstract:
Same conventions as chat_evals.py / chat_math_extra.py: the model's chat template, the scored text = what follows the LAST </think>
(an unclosed block gives no answer), `\\boxed{}` strict and lenient numbers and pass@k over the samples (chat_metrics.chat_math_metric).

polymath_chat: the question in its own language, followed by the OFFICIAL answer-format note of the PolyMath repo for that language
(instruction.py `query_dic`, "... $\\boxed{}$"). No language forcing (instruction.py `language_control` is not used). The gold is the
dataset's `answer`, wrapped in $...$ when it is bare (the extractor reads LaTeX). Scoring: math-verify (the fork's math extractor, as
math_500_chat), not the official `math_equal` of eval/scripts.py (the official takes the FIRST box; this task takes the math-verify
match of the strict box). Subtasks polymath_chat:<level>:<lang>, level in low / medium / high / top, lang = the dataset config.
Subset: a fixed seeded draw of 25 problems per (language, level) = 1,800 items (chat_polymath_subsets.json, seed chat_polymath-v1, drawn
by `python -m lighteval.tasks.tasks.chat_polymath --draw`). The full set (9,000 items) is the same tasks without the id filter, to be added as a separate task name later.
Metrics: avg@n (primary), avg@n_lenient, pass@k, and dw_acc = the official difficulty-weighted average (weights 1 / 2 / 4 / 8 for
low / medium / high / top, divided by 15). dw_acc is stored per item as correct x 4 w_level / 15, so its mean over the balanced task
(each level 25 x 18 items) IS the official weighted average, and the registry's macro mean over the subtasks is the plain mean of the
four level accuracies (the primary). The per-subtask dw_acc of one level is that level's value times 4 w / 15 (read the avg@n of a level
subtask for its accuracy).

mgsm_chat: the question in its own language (Global-MGSM), the English chat math instruction (MATH_PROMPT, "\\boxed{ANSWER}") as in
fusion_aya_math_chat, 250 items per language, all 40 languages of the base task, avg@n with one sample (as gsm8k_chat).

languages:
polymath_chat: ar, bn, de, en, es, fr, id, it, ja, ko, ms, pt, ru, sw, te, th, vi, zh (18)
mgsm_chat: am, ar, bn, ca, cs, cy, de, el, es, eu, fr, gl, gu, ha, hu, ja, km, kn, ko, ky, lg, my, ne, ru, si, sn, sr, st, sw, ta, te, th, ur, uz, vi, wo, xh, yo, zh, zu (40)

tags:
chat, reasoning, math, multilingual

paper:
https://arxiv.org/abs/2504.18428 (PolyMath); https://arxiv.org/abs/2210.03057 (MGSM)
"""

import json
import random
from functools import partial
from pathlib import Path

from lighteval.metrics.chat_metrics import _grouping, _math_extractor, _math_scores, chat_math_metric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.chat_evals import BUDGET, MATH_PROMPT


SEED = "chat_polymath-v1"
SUBSETS_FILE = Path(__file__).with_name("chat_polymath_subsets.json")
PER_CELL = 25          # problems per (language, level) drawn by draw() (the task reads the stored ids; the full 125 = a task without the filter)
N_POLY = 4             # samples per problem (as math_500_chat and the fusion task); the run's gen.n replaces it
N_MGSM = 1             # one sample, as gsm8k_chat

# ----------------------------------------------------------------------------------------------------------------- PolyMath
POLYMATH_REPO = "Qwen/PolyMath"
POLYMATH_REVISION = "71e0db902e0ece05e208dfd8b1695bd3d95cf130"   # main on 2026-11-14 (the card and its configs)
POLYMATH_LANGUAGES = ("ar", "bn", "de", "en", "es", "fr", "id", "it", "ja", "ko", "ms", "pt", "ru", "sw", "te", "th", "vi", "zh")
POLYMATH_LEVELS = ("low", "medium", "high", "top")     # the dataset's split names (the repo's eval script says "middle")
# the official weights of the difficulty-weighted average (eval/run_eval.py: sum over levels 2^i acc_i / 15, i = 0..3 for low..top)
LEVEL_WEIGHT = {"low": 1, "medium": 2, "high": 4, "top": 8}
DW_DENOM = 15
# instruction.py `query_dic`: the answer-format note of the official repo, one per language, appended to the question
POLYMATH_NOTE = {
    "en": "Note: Please put the final answer in the $\\boxed{}$.",
    "zh": "注意：请将最终答案放在 $\\boxed{}$ 中。",
    "ar": "ملاحظة: يُرجى وضع الإجابة النهائية في $\\boxed{}$.",
    "bn": "বিঃদ্রঃ: অনুগ্রহ করে চূড়ান্ত উত্তরটি $\\boxed{}$ এর মধ্যে রাখুন।",
    "de": "Hinweis: Bitte setzen Sie die endgültige Antwort in $\\boxed{}$.",
    "es": "Nota: Por favor, coloque la respuesta final en el $\\boxed{}$.",
    "fr": "Remarque : Veuillez mettre la réponse finale dans le $\\boxed{}$.",
    "id": "Catatan: Silakan letakkan jawaban akhir di dalam $\\boxed{}$.",
    "it": "Nota: Per favore, metti la risposta finale nel $\\boxed{}$.",
    "ja": "注意：最終的な答えを $\\boxed{}$ に入れてください。",
    "ko": "참고: 최종 답안을 $\\boxed{}$ 안에 넣어 주세요.",
    "ms": "Nota: Sila letakkan jawapan akhir dalam $\\boxed{}$.",
    "pt": "Nota: Por favor, coloque a resposta final no $\\boxed{}$.",
    "ru": "Примечание: Пожалуйста, поместите окончательный ответ в $\\boxed{}$.",
    "sw": "Kumbuka: Tafadhali weka jibu la mwisho katika $\\boxed{}$.",
    "te": "గమనిక: దయచేసి తుది జవాబును $\\boxed{}$ లో ఉంచండి.",
    "th": "หมายเหตุ: กรุณาใส่คำตอบสุดท้ายใน $\\boxed{}$.",
    "vi": "Lưu ý: Vui lòng đặt câu trả lời cuối cùng trong $\\boxed{}$.",
}

# ------------------------------------------------------------------------------------------------------------------ MGSM
MGSM_REPO = "CohereLabs/global-mgsm"
MGSM_REVISION = "29087245a1a9788d4e0413f4927e994d49633bbb"      # the pin of the base mgsm:gen task (configs/datasets.lock.yaml)
MGSM_LANGUAGES = (
    "am", "ar", "bn", "ca", "cs", "cy", "de", "el", "es", "eu", "fr", "gl", "gu", "ha", "hu", "ja", "km", "kn", "ko", "ky",
    "lg", "my", "ne", "ru", "si", "sn", "sr", "st", "sw", "ta", "te", "th", "ur", "uz", "vi", "wo", "xh", "yo", "zh", "zu",
)


def _bare_gold(answer) -> str:
    """The extractor reads LaTeX: a bare answer is wrapped in $...$ (as the HMMT gold of chat_evals); an answer already in $...$ is kept."""
    text = str(answer).strip()
    return text if "$" in text else f"${text}$"


def polymath_prompt(line, task_name: str = None, language: str = None, level: str = None):
    return Doc(
        task_name=task_name,
        query=f"{line['question'].strip()}\n\n{POLYMATH_NOTE[language]}",
        choices=[_bare_gold(line["answer"])],
        gold_index=0,
        specific={"id": line["id"], "language": language, "level": level},
    )


def _polymath_scores_fn():
    """The math scores of chat_metrics (strict avg@n, lenient avg@n_lenient) plus the per-item weighted value of dw_acc."""
    base = _math_scores(_math_extractor())

    def score(doc, pred):
        s = base(doc, pred)
        weight = 4 * LEVEL_WEIGHT[doc.specific["level"]] / DW_DENOM
        return {"avg@n": s["avg@n"], "avg@n_lenient": s["avg@n_lenient"], "dw_acc": s["avg@n"] * weight}

    return score


def polymath_metric(n: int):
    return _grouping(n, _polymath_scores_fn(), ("avg@n", "avg@n_lenient", "dw_acc"))


def oracle_box(answer) -> str:
    """The box a correct model would write for this gold: the gold text without its outer $...$ inside \\boxed{}."""
    text = str(answer).strip()
    if text.startswith("$") and text.endswith("$") and text.count("$") == 2:
        text = text[1:-1]
    return f"\\boxed{{{text}}}"


def gold_is_scorable(answer) -> bool:
    """Does the scorer give 1 to the oracle box of this gold? 17 of the 9,000 golds fail it (multi-answer prose such as
    "$2^{2^l}$ for $l=1,2,4,8$", a broken "$b \\geq $-\\frac{2}{3}$", "P(x)=c(x^2+3) for any constant c"): no model answer can
    match them, so draw() leaves them out of the pool before the seeded draw (a data fix; the full set keeps them)."""
    doc = Doc(query="", choices=[_bare_gold(answer)], gold_index=0, specific={"level": "low"})
    return _polymath_scores_fn()(doc, oracle_box(answer))["avg@n"] == 1.0


def polymath_filter(ids: frozenset):
    def keep(row) -> bool:
        return row["id"] in ids

    return keep


def mgsm_chat_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT.format(problem=line["question"].strip()),
        choices=[_bare_gold(line["answer"])],
        gold_index=0,
    )


# ----------------------------------------------------------------------------------------------------------------- tasks
def _load_subsets() -> dict:
    return json.loads(SUBSETS_FILE.read_text())


def _build_tasks():
    tasks = []
    subsets = _load_subsets()["polymath"] if SUBSETS_FILE.exists() else None
    for language in POLYMATH_LANGUAGES:
        for level in POLYMATH_LEVELS:
            ids = frozenset(subsets[language][level]) if subsets else None
            tasks.append(
                LightevalTaskConfig(
                    name=f"polymath_chat:{level}:{language}",
                    prompt_function=partial(polymath_prompt, language=language, level=level),
                    hf_repo=POLYMATH_REPO,
                    hf_subset=language,
                    hf_revision=POLYMATH_REVISION,
                    hf_avail_splits=[level],
                    evaluation_splits=[level],
                    few_shots_split=None,
                    few_shots_select=None,
                    generation_size=BUDGET,
                    stop_sequence=[],
                    hf_filter=polymath_filter(ids) if ids is not None else None,
                    metrics=[polymath_metric(N_POLY)],
                    version=0,
                )
            )
    for language in MGSM_LANGUAGES:
        tasks.append(
            LightevalTaskConfig(
                name=f"mgsm_chat:{language}",
                prompt_function=mgsm_chat_prompt,
                hf_repo=MGSM_REPO,
                hf_subset=language,
                hf_revision=MGSM_REVISION,
                hf_avail_splits=["test"],
                evaluation_splits=["test"],
                few_shots_split=None,
                few_shots_select=None,
                generation_size=BUDGET,
                stop_sequence=[],
                metrics=[chat_math_metric(N_MGSM)],
                version=0,
            )
        )
    return tasks


TASKS_TABLE = _build_tasks()


# ----------------------------------------------------------------------------------------------------------------- the draw
def draw(out: Path = SUBSETS_FILE, per_cell: int = PER_CELL) -> dict:
    """Draw the seeded subsets from the pinned dataset and write them (run once on the login node; the file is the record).
    Seed: random.Random(f"{SEED}:<language>:<level>") over the SORTED ids of the cell, so the draw is the same on every machine."""
    from datasets import load_dataset

    data = {}
    for language in POLYMATH_LANGUAGES:
        data[language] = {}
        for level in POLYMATH_LEVELS:
            ds = load_dataset(POLYMATH_REPO, language, split=level, revision=POLYMATH_REVISION)
            ids = sorted({str(r["id"]) for r in ds if gold_is_scorable(r["answer"])})
            data[language][level] = sorted(random.Random(f"{SEED}:{language}:{level}").sample(ids, min(per_cell, len(ids))))
    result = {"seed": SEED, "per_cell": per_cell, "repo": POLYMATH_REPO, "revision": POLYMATH_REVISION,
              "pool": "problems whose gold passes gold_is_scorable (the oracle box scores 1)", "polymath": data}
    out.write_text(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    import sys

    if "--draw" in sys.argv:
        d = draw()
        print(f"polymath languages {len(d['polymath'])}, items {sum(len(v) for lang in d['polymath'].values() for v in lang.values())}")
    else:
        print("usage: python -m lighteval.tasks.tasks.chat_polymath --draw")
