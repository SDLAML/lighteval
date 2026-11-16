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
Chat evals, PolyMath and MGSM (titan_eval, post-trained models): polymath_chat (17 non-English languages x 4 difficulty levels), polymath_en_chat (the English problems), mgsm_chat (40 languages, no English)

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
FULL set (owner rule 11-14: no seeded subsets to save compute): all 125 problems of every (language, level) cell. polymath_chat = the 17 NON-English languages = 8,500
problems (English is polymath_en_chat, owner 11-14: a multilingual task excludes English when English runs); all 18 languages = 9,000. The 90 golds (85 non-English) that the
scorer cannot read (the oracle box of the gold scores 0: multi-answer prose such as "$2^{2^l}$ for $l=1,2,4,8$" and "$P(x)=c(x^2+3)$ for any constant $c$", a broken
"$b \\geq $-\\frac{2}{3}$", a missing brace; the same 5 problems in every language: 2 medium + 3 top) STAY IN the set and are COUNTED: the metric `gold_unscorable` is the
share of problems whose gold the scorer cannot read (0.0100 = 85 / 8,500 here, 5 / 500 for English), so the ceiling of any model is 1 - gold_unscorable (99.0 %). They cost every
model the same and are not removed.
polymath_en_chat: the ENGLISH problems only (4 levels x 125 = 500 problems), the same prompt, scorer, level subtasks (polymath_en_chat:<level>), primary and extra metrics.
The opt-in flag `data.sample_per_group=N` (titan_eval) keeps a seeded N per (language, level) cell (seed chat_polymath-v2:<language>:<level>, drawn by lighteval.tasks.sampling
from all 125 problems of the cell, nothing stored).
Metrics: avg@n (primary), avg@n_lenient, pass@k, gold_unscorable (diagnostic, lower is better), and dw_acc = the official difficulty-weighted average (weights 1 / 2 / 4 / 8 for
low / medium / high / top, divided by 15). dw_acc is stored per item as correct x 4 w_level / 15, so its mean over the balanced task
(each level 125 x 17 items in polymath_chat, 125 in polymath_en_chat) IS the official weighted average, and the registry's macro mean over the subtasks is the plain mean of the
four level accuracies (the primary). The per-subtask dw_acc of one level is that level's value times 4 w / 15 (read the avg@n of a level
subtask for its accuracy).

mgsm_chat: the OFFICIAL MGSM protocol of OpenAI simple-evals `mgsm_eval.py` (owner 11-16, B4; until then the English chat math instruction with "\\boxed{ANSWER}" and the math checker):
0-shot, ONE user message = the dataset's own per-language `instruction` (Global-MGSM column; for the 10 languages of simple-evals it is that file's text, checked) + a blank line + the question in its
own language. The answer is the LAST number after the LAST occurrence of the dataset's `answer_prefix` ("Answer" / "Antwort" / ...) in the scored text (simple-evals `parse_answer`, l.83-94:
`answer.split(answer_prefix)[-1]`, commas removed, `\\d+\\.?\\d*`, the last number, a final "." dropped; no prefix = no answer), compared as a string with the gold after `score_mgsm` (l.97-104: a
trailing ".0" / zeros of a decimal are dropped, commas removed). Metrics: avg@n (primary, exact match, n = 1 sample as gsm8k_chat) and no_answer_prefix (diagnostic, lower is better: the share
of samples without the prefix, which score 0). The scored text is what follows the LAST </think> (an unclosed block gives no answer). 250 items per language, all 40 languages of the base
task (no English: English GSM is gsm8k_chat).
The opt-in flag `data.sample_per_group=N` keeps a seeded N per language (seed chat_polymath-v2:mgsm:<lang>, key = the question text).

languages:
polymath_chat: ar, bn, de, es, fr, id, it, ja, ko, ms, pt, ru, sw, te, th, vi, zh (17; en = polymath_en_chat)
mgsm_chat: am, ar, bn, ca, cs, cy, de, el, es, eu, fr, gl, gu, ha, hu, ja, km, kn, ko, ky, lg, my, ne, ru, si, sn, sr, st, sw, ta, te, th, ur, uz, vi, wo, xh, yo, zh, zu (40)

tags:
chat, reasoning, math, multilingual

paper:
https://arxiv.org/abs/2504.18428 (PolyMath); https://arxiv.org/abs/2210.03057 (MGSM)
"""

import re
from functools import lru_cache, partial

from lighteval.metrics.chat_metrics import _grouping, _math_extractor, _math_scores
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.sampling import subset_filter
from lighteval.tasks.tasks.chat_evals import BUDGET


SEED = "chat_polymath-v2"      # seed prefix of the opt-in subsets (data.sample_per_group); the full sets use no seed. v2: the draw is over all 125 problems of a cell (v1 left out the unscorable golds)
PER_CELL = 125         # problems per (language, level) in the dataset
N_POLY = 4             # samples per problem (as math_500_chat and the fusion task); the run's gen.n replaces it
N_MGSM = 1             # one sample, as gsm8k_chat

# ----------------------------------------------------------------------------------------------------------------- PolyMath
POLYMATH_REPO = "Qwen/PolyMath"
POLYMATH_REVISION = "71e0db902e0ece05e208dfd8b1695bd3d95cf130"   # main on 2026-11-14 (the card and its configs)
POLYMATH_LANGUAGES = ("ar", "bn", "de", "en", "es", "fr", "id", "it", "ja", "ko", "ms", "pt", "ru", "sw", "te", "th", "vi", "zh")
POLYMATH_MULTILINGUAL = tuple(l for l in POLYMATH_LANGUAGES if l != "en")     # polymath_chat: every language but English (owner 11-14: a multilingual task excludes English when English runs; English = polymath_en_chat)
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


@lru_cache(maxsize=None)
def _oracle_scores_fn():
    return _math_scores(_math_extractor())


def _polymath_scores_fn():
    """The math scores of chat_metrics (strict avg@n, lenient avg@n_lenient), the per-item weighted value of dw_acc and `gold_unscorable` (1.0 when the
    scorer cannot read the item's own gold: the oracle box of the gold scores 0, see gold_is_scorable)."""
    base = _oracle_scores_fn()

    def score(doc, pred):
        s = base(doc, pred)
        weight = 4 * LEVEL_WEIGHT[doc.specific["level"]] / DW_DENOM
        return {"avg@n": s["avg@n"], "avg@n_lenient": s["avg@n_lenient"], "dw_acc": s["avg@n"] * weight,
                "gold_unscorable": 0.0 if _gold_scorable(str(doc.get_golds()[0])) else 1.0}

    return score


def polymath_metric(n: int):
    metric = _grouping(n, _polymath_scores_fn(), ("avg@n", "avg@n_lenient", "dw_acc", "gold_unscorable"))
    metric.higher_is_better["gold_unscorable"] = False        # a diagnostic: the share of problems nobody can score
    return metric


def oracle_box(answer) -> str:
    """The box a correct model would write for this gold: the gold text without its outer $...$ inside \\boxed{}."""
    text = str(answer).strip()
    if text.startswith("$") and text.endswith("$") and text.count("$") == 2:
        text = text[1:-1]
    return f"\\boxed{{{text}}}"


@lru_cache(maxsize=None)
def _gold_scorable(bare_gold: str) -> bool:
    """Does the scorer give 1 to the oracle box of this (already wrapped) gold? Cached per gold text: the same problems recur in every language."""
    text = bare_gold.strip()
    doc = Doc(query="", choices=[bare_gold], gold_index=0, specific={"level": "low"})
    return _oracle_scores_fn()(doc, oracle_box(text))["avg@n"] == 1.0


def gold_is_scorable(answer) -> bool:
    """Does the scorer give 1 to the oracle box of this gold? The golds that fail it (multi-answer prose such as "$2^{2^l}$ for $l=1,2,4,8$", a broken
    "$b \\geq $-\\frac{2}{3}$", "P(x)=c(x^2+3) for any constant c") cannot be matched by any model answer: they stay in the set and are counted (metric gold_unscorable)."""
    return _gold_scorable(_bare_gold(answer))


def _poly_key(row) -> str:
    return str(row["id"])


def _mgsm_key(row) -> str:
    return str(row["question"])


def mgsm_parse_answer(answer: str, answer_prefix: str) -> str:
    """simple-evals `parse_answer` (mgsm_eval.py l.83-94, github.com/openai/simple-evals @ 652c89d0), unchanged: no prefix -> ""; else the last number after the LAST prefix, commas removed,
    a final "." dropped."""
    if answer_prefix not in answer:
        return ""
    answer_text = answer.split(answer_prefix)[-1].strip()
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))
    return numbers[-1].rstrip(".") if numbers else ""


def mgsm_score_one(target: str, prediction: str) -> bool:
    """simple-evals `score_mgsm` (mgsm_eval.py l.97-104), unchanged: zeros of a decimal prediction are dropped, commas removed on both sides, string equality."""
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")
    target = target.replace(",", "")
    prediction = prediction.replace(",", "")
    return target == prediction


def _mgsm_scores(doc: Doc, pred: str) -> dict[str, float]:
    if not pred or not pred.strip():
        return {"avg@n": 0.0, "no_answer_prefix": 1.0}
    prefix = doc.specific["answer_prefix"]
    return {"avg@n": float(mgsm_score_one(doc.get_golds()[0], mgsm_parse_answer(pred, prefix))), "no_answer_prefix": float(prefix not in pred)}


def mgsm_metric(n: int):
    metric = _grouping(n, _mgsm_scores, ("avg@n", "no_answer_prefix"))
    metric.higher_is_better["no_answer_prefix"] = False       # a diagnostic: the share of samples that never wrote the answer prefix
    return metric


def mgsm_chat_prompt(line, task_name: str = None):
    """The dataset's own prompt: `instruction` (per language) + blank line + `question`; the gold is the dataset's `answer` (an integer string). Global-MGSM card: "build the prompt by
    concatenating instruction and question"; simple-evals: `instruction.format(input=question)` with the instruction ending "\\n\\n{input}"."""
    return Doc(
        task_name=task_name,
        query=f"{line['instruction'].strip()}\n\n{line['question'].strip()}",
        choices=[str(line["answer"]).strip()],
        gold_index=0,
        specific={"answer_prefix": line["answer_prefix"]},
    )


# ----------------------------------------------------------------------------------------------------------------- tasks
def _polymath_config(name: str, language: str, level: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name,
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
        hf_filter=subset_filter(f"{SEED}:{language}:{level}", _poly_key, [level]),        # None = the full cell (125 problems)
        metrics=[polymath_metric(N_POLY)],
        version=0,
    )


def _build_tasks():
    tasks = []
    for language in POLYMATH_MULTILINGUAL:
        for level in POLYMATH_LEVELS:
            tasks.append(_polymath_config(f"polymath_chat:{level}:{language}", language, level))
    for level in POLYMATH_LEVELS:                      # polymath_en_chat: the English problems only (owner 11-14: PolyMath in Math and in Multilingual)
        tasks.append(_polymath_config(f"polymath_en_chat:{level}", "en", level))
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
                hf_filter=subset_filter(f"{SEED}:mgsm:{language}", _mgsm_key, ["test"]),       # None = the full 250 problems
                metrics=[mgsm_metric(N_MGSM)],
                version=1,          # 0 -> 1 (owner 11-16, B4): the dataset's own instruction + answer prefix, simple-evals parse / score (was: English MATH_PROMPT + \\boxed, math checker)
            )
        )
    return tasks


TASKS_TABLE = _build_tasks()
