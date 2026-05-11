"""
name:
Mgsm

dataset:
CohereLabs/global-mgsm

abstract:
MGSM (Multilingual Grade School Math) is a multilingual benchmark testing
mathematical reasoning across languages, derived from GSM8K.

Refactored to use unified :gen suffix consistent with English gsm8k.py.
English is excluded — use gsm8k.py for English evaluation.
Reports both expr_gold_metric (math expression parser) and
MultilingualQuasiExactMatchMetric (language-aware fuzzy match, handles
non-ASCII digit systems like Japanese/Thai).

languages:
amharic, arabic, bengali, catalan, czech, welsh, german, greek, spanish,
basque, french, galician, gujarati, hausa, hungarian, japanese, khmer,
kannada, korean, kyrgyz, ganda, burmese, nepali, russian, sinhala, shona,
serbian, southern sotho, swahili, tamil, telugu, thai, urdu, uzbek,
vietnamese, wolof, xhosa, yoruba, chinese, zulu

tags:
math, multilingual, reasoning

paper:
https://arxiv.org/abs/2210.03057
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import MultilingualQuasiExactMatchMetric
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


# Languages covered by CohereLabs/global-mgsm (English excluded — use gsm8k.py)
_LANGUAGES = [
    Language.AMHARIC,
    Language.ARABIC,
    Language.BENGALI,
    Language.CATALAN,
    Language.CZECH,
    Language.WELSH,
    Language.GERMAN,
    Language.GREEK,
    Language.SPANISH,
    Language.BASQUE,
    Language.FRENCH,
    Language.GALICIAN,
    Language.GUJARATI,
    Language.HAUSA,
    Language.HUNGARIAN,
    Language.JAPANESE,
    Language.KHMER,
    Language.KANNADA,
    Language.KOREAN,
    Language.KIRGHIZ,
    Language.GANDA,
    Language.BURMESE,
    Language.NEPALI,
    Language.RUSSIAN,
    Language.SINHALA,
    Language.SHONA,
    Language.SERBIAN,
    Language.SOUTHERN_SOTHO,
    Language.SWAHILI,
    Language.TAMIL,
    Language.TELUGU,
    Language.THAI,
    Language.URDU,
    Language.UZBEK,
    Language.VIETNAMESE,
    Language.WOLOF,
    Language.XHOSA,
    Language.YORUBA,
    Language.CHINESE,
    Language.ZULU,
]


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mgsm:{language.value}:gen",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [line["answer"]],
            },
        ),
        hf_repo="CohereLabs/global-mgsm",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        few_shots_split=None,
        generation_size=512,
        metrics=[
            Metrics.expr_gold_metric,
            MultilingualQuasiExactMatchMetric(language, "full"),
        ],
        stop_sequence=["Question:", "Answer:"],
    )
    for language in _LANGUAGES
]
