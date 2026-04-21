"""
name:
Mgsm

dataset:
juletxara/mgsm

abstract:
MGSM (Multilingual Grade School Math) is a multilingual benchmark testing
mathematical reasoning across languages, derived from GSM8K.

Refactored to use unified :gen suffix consistent with English gsm8k.py.
English is excluded — use gsm8k.py for English evaluation.
Reports both expr_gold_metric (math expression parser) and
MultilingualQuasiExactMatchMetric (language-aware fuzzy match, handles
non-ASCII digit systems like Japanese/Thai).

tags:
math, multilingual, reasoning

paper:
https://arxiv.org/abs/2210.03057
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import MultilingualQuasiExactMatchMetric
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


_LANGUAGES = [
    # Language.ENGLISH,
    Language.ARABIC,
    Language.CZECH,
    Language.GREEK,
    Language.BASQUE,
    Language.KOREAN,
    # Language.SERBIAN,
    Language.CATALAN,
    Language.GALICIAN,
    Language.HUNGARIAN,
    Language.VIETNAMESE,
    Language.SPANISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.RUSSIAN,
    Language.CHINESE,
    Language.JAPANESE,
    Language.THAI,
    # Language.SWAHILI,
    # Language.BENGALI,
    # Language.TELUGU,
]


# CoT prompt templates per language, mirroring the English template used for
# the MATH task. Each template must expose a single `{prompt}` placeholder and
# end with the equivalent of "Solution:" so the model continues with the
# step-by-step reasoning in the target language.
MGSM_COT_PROMPT_TEMPLATES: dict[Language, str] = {
    Language.ENGLISH: (
        "Solve the following math problem. Think step by step before giving the final answer.\n\n"
        "Problem:\n{prompt}\n\nSolution:"
    ),
    Language.ARABIC: (
        "حل المسألة الرياضية التالية. فكّر خطوة بخطوة قبل تقديم الإجابة النهائية.\n\n"
        "المسألة:\n{prompt}\n\nالحل:"
    ),
    Language.CZECH: (
        "Vyřeš následující matematickou úlohu. Než uvedeš konečnou odpověď, přemýšlej krok za krokem.\n\n"
        "Úloha:\n{prompt}\n\nŘešení:"
    ),
    Language.GREEK: (
        "Λύσε το παρακάτω μαθηματικό πρόβλημα. Σκέψου βήμα προς βήμα πριν δώσεις την τελική απάντηση.\n\n"
        "Πρόβλημα:\n{prompt}\n\nΛύση:"
    ),
    Language.BASQUE: (
        "Ebatzi honako matematikako problema hau. Pentsatu pausoz pauso azken erantzuna eman aurretik.\n\n"
        "Problema:\n{prompt}\n\nEbazpena:"
    ),
    Language.KOREAN: (
        "다음 수학 문제를 풀어라. 최종 답을 제시하기 전에 단계별로 생각하라.\n\n"
        "문제:\n{prompt}\n\n풀이:"
    ),
    Language.CATALAN: (
        "Resol el següent problema de matemàtiques. Pensa pas a pas abans de donar la resposta final.\n\n"
        "Problema:\n{prompt}\n\nSolució:"
    ),
    Language.GALICIAN: (
        "Resolve o seguinte problema matemático. Pensa paso a paso antes de dar a resposta final.\n\n"
        "Problema:\n{prompt}\n\nSolución:"
    ),
    Language.HUNGARIAN: (
        "Oldd meg a következő matematikai feladatot. Gondolkodj lépésről lépésre, mielőtt megadnád a végső választ.\n\n"
        "Feladat:\n{prompt}\n\nMegoldás:"
    ),
    Language.VIETNAMESE: (
        "Giải bài toán sau đây. Hãy suy nghĩ từng bước trước khi đưa ra câu trả lời cuối cùng.\n\n"
        "Bài toán:\n{prompt}\n\nLời giải:"
    ),
    Language.SPANISH: (
        "Resuelve el siguiente problema de matemáticas. Piensa paso a paso antes de dar la respuesta final.\n\n"
        "Problema:\n{prompt}\n\nSolución:"
    ),
    Language.FRENCH: (
        "Résous le problème de mathématiques suivant. Réfléchis étape par étape avant de donner la réponse finale.\n\n"
        "Problème :\n{prompt}\n\nSolution:"
    ),
    Language.GERMAN: (
        "Löse die folgende Mathematikaufgabe. Denke Schritt für Schritt nach, bevor du die endgültige Antwort gibst.\n\n"
        "Aufgabe:\n{prompt}\n\nLösung:"
    ),
    Language.RUSSIAN: (
        "Реши следующую математическую задачу. Рассуждай пошагово, прежде чем дать окончательный ответ.\n\n"
        "Задача:\n{prompt}\n\nРешение:"
    ),
    Language.CHINESE: (
        "解答下面的数学题。在给出最终答案前，请一步一步思考。\n\n"
        "问题：\n{prompt}\n\n解答："
    ),
    Language.JAPANESE: (
        "次の数学の問題を解きなさい。最終的な答えを出す前に、順を追って考えなさい。\n\n"
        "問題：\n{prompt}\n\n解答："
    ),
    Language.THAI: (
        "จงแก้โจทย์คณิตศาสตร์ต่อไปนี้ คิดทีละขั้นตอนก่อนให้คำตอบสุดท้าย\n\n"
        "โจทย์:\n{prompt}\n\nวิธีทำ:"
    ),
}


def mgsm_cot_prompt(language: Language):
    """Build a CoT prompt function for MGSM in the given language."""
    template = MGSM_COT_PROMPT_TEMPLATES[language]

    def prompt_fn(line, task_name: str = None):
        return Doc(
            task_name=task_name,
            query=template.format(prompt=line["question"]),
            choices=[f" {line['answer']}"],
            gold_index=0,
        )

    return prompt_fn


def mgsm_qa_prompt(language: Language):
    """Original `Question:`/`Answer:` style prompt (language-localized labels,
    no CoT instruction)."""
    return get_qa_prompt_function(
        language,
        lambda line: {
            "question": line["question"],
            "choices": [str(line["answer"])],
        },
    )


def _mgsm_task(language: Language, *, suffix: str, prompt_function, generation_size: int, stop_sequence: list[str]):
    return LightevalTaskConfig(
        name=f"mgsm_{language.value}:{suffix}",
        prompt_function=prompt_function,
        hf_repo="CohereLabs/global-mgsm",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        generation_size=generation_size,
        metrics=[
            Metrics.expr_gold_metric,
            # MultilingualQuasiExactMatchMetric(language, "full"),
        ],
        stop_sequence=stop_sequence,
    )


# Default (`:gen`) — original QA-style prompt.
_QA_TASKS = [
    _mgsm_task(
        language,
        suffix="gen",
        prompt_function=mgsm_qa_prompt(language),
        generation_size=512,
        stop_sequence=["Question:", "Problem:", "\n\n"],
    )
    for language in _LANGUAGES
]

# CoT (`:cot`) — step-by-step prompt in the target language.
_COT_TASKS = [
    _mgsm_task(
        language,
        suffix="cot",
        prompt_function=mgsm_cot_prompt(language),
        generation_size=1024,
        stop_sequence=["Question:", "Problem:"],
    )
    for language in _LANGUAGES
]


TASKS_TABLE = _QA_TASKS + _COT_TASKS