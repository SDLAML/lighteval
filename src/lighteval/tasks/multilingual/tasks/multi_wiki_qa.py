"""
name:
Multi Wiki QA

dataset:
alexandrainst/multi-wiki-qa

abstract:
MultiWikiQA is a reading comprehension benchmark in 300+ languages based on
Wikipedia articles coupled with LLM-generated questions (Gemini-1.5-pro) and
extractive answers. The dataset uses the standard SQuAD format with context,
question, and answer spans. Questions are rephrased to prevent word-matching
shortcuts. Each language subset contains up to 5,000 samples.

languages:
arabic, bashkir, basque, belarusian, bengali, bulgarian, catalan, cebuano,
chinese, croatian, czech, danish, dutch, english, estonian, finnish, french,
galician, german, greek, gujarati, hindi, hungarian, icelandic, indonesian,
italian, japanese, korean, macedonian, norwegian, polish, portuguese, punjabi,
romanian, russian, serbian, shan, slovak, spanish, swahili, swedish, tagalog,
tamil, tatar, telugu, thai, turkish, udmurt, ukrainian, urdu, uzbek, vietnamese

tags:
multilingual, qa, reading-comprehension

paper:
https://arxiv.org/abs/2509.04111
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    MultilingualQuasiExactMatchMetric,
    MultilingualQuasiF1ScoreMetric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


def _make_qa_adapter():
    def adapter(line):
        choices = [ans for ans in line["answers"]["text"] if len(ans) > 0]
        if not choices:
            return None
        return {
            "question": line["question"],
            "context": line["context"],
            "choices": choices,
        }
    return adapter


# Languages that have full QA template support (question_word + answer translations)
# and a matching subset in the dataset via standardize_tag(lang.value).
_STANDARD_LANGUAGES = [
    Language.ARABIC,
    Language.BASHKIR,
    Language.BASQUE,
    Language.BELARUSIAN,
    Language.BENGALI,
    Language.BULGARIAN,
    Language.CATALAN,
    Language.CEBUANO,
    Language.CROATIAN,
    Language.CZECH,
    Language.DANISH,
    Language.DUTCH,
    Language.ENGLISH,
    Language.ESTONIAN,
    Language.FINNISH,
    Language.FRENCH,
    Language.GALICIAN,
    Language.GERMAN,
    Language.GREEK,
    Language.GUJARATI,
    Language.HINDI,
    Language.HUNGARIAN,
    Language.ICELANDIC,
    Language.INDONESIAN,
    Language.ITALIAN,
    Language.JAPANESE,
    Language.KOREAN,
    Language.MACEDONIAN,
    Language.NORWEGIAN,
    Language.POLISH,
    Language.PUNJABI,
    Language.ROMANIAN,
    Language.RUSSIAN,
    Language.SERBIAN,
    Language.SHAN,
    Language.SLOVAK,
    Language.SPANISH,
    Language.SWAHILI,
    Language.SWEDISH,
    Language.TAGALOG,
    Language.TAMIL,
    Language.TATAR,
    Language.TELUGU,
    Language.THAI,
    Language.TURKISH,
    Language.UDMURT,
    Language.UKRAINIAN,
    Language.URDU,
    Language.UZBEK,
    Language.VIETNAMESE,
]

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"multi_wiki_qa_{standardize_tag(language.value)}",
        prompt_function=get_qa_prompt_function(language, _make_qa_adapter()),
        hf_repo="alexandrainst/multi-wiki-qa",
        hf_subset=standardize_tag(language.value),
        evaluation_splits=["train",],
        hf_avail_splits=["train"],
        generation_size=100,
        stop_sequence=["\n",],
        metrics=[
            MultilingualQuasiExactMatchMetric(language, "prefix"),
            MultilingualQuasiF1ScoreMetric(language),
        ],
    )
    for language in _STANDARD_LANGUAGES
]

# Chinese: the dataset splits Simplified (zh-cn) and Traditional (zh-tw) Mandarin
for _zh_subset in ["zh-cn", "zh-tw"]:
    _zh_name = _zh_subset.replace("-", "_")
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"multi_wiki_qa_{_zh_name}",
            prompt_function=get_qa_prompt_function(Language.CHINESE, _make_qa_adapter()),
            hf_repo="alexandrainst/multi-wiki-qa",
            hf_subset=_zh_subset,
            evaluation_splits=["train",],
            hf_avail_splits=["train"],
            generation_size=100,
            stop_sequence=["\n",],
            metrics=[
                MultilingualQuasiExactMatchMetric(Language.CHINESE, "prefix"),
                MultilingualQuasiF1ScoreMetric(Language.CHINESE),
            ],
        )
    )

# Portuguese: the dataset splits European (pt-pt) and Brazilian (pt-br) Portuguese
for _pt_subset in ["pt-pt", "pt-br"]:
    _pt_name = _pt_subset.replace("-", "_")
    TASKS_TABLE.append(
        LightevalTaskConfig(
            name=f"multi_wiki_qa_{_pt_name}",
            prompt_function=get_qa_prompt_function(Language.PORTUGUESE, _make_qa_adapter()),
            hf_repo="alexandrainst/multi-wiki-qa",
            hf_subset=_pt_subset,
            evaluation_splits=["train",],
            hf_avail_splits=["train"],
            generation_size=100,
            stop_sequence=["\n",],
            metrics=[
                MultilingualQuasiExactMatchMetric(Language.PORTUGUESE, "prefix"),
                MultilingualQuasiF1ScoreMetric(Language.PORTUGUESE),
            ],
        )
    )
