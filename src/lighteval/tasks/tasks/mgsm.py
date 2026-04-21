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


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mgsm_{language.value}:gen",
        prompt_function=get_qa_prompt_function(
            language,
            lambda line: {
                "question": line["question"],
                "choices": [str(line["answer"])],
            },
        ),
        hf_repo="CohereLabs/global-mgsm",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=("test",),
        generation_size=512,
        metrics=[
            Metrics.expr_gold_metric,
            # MultilingualQuasiExactMatchMetric(language, "full"),
        ],
        stop_sequence=["Question:", "Problem:", "\n\n"],
    )
    for language in _LANGUAGES
]