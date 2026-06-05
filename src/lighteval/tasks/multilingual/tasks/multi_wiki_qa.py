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

Naming follows the multilingual convention `multi_wiki_qa:{lang}:gen`. English
is intentionally kept (this task has no dedicated English file), so
`multi_wiki_qa:eng:gen` is runnable like any other language.

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
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
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


def _make_bpb_prompt(language: Language):
    """Reuse the gen QA query, but score only the first gold continuation (BPB)."""
    base = get_qa_prompt_function(language, _make_qa_adapter())

    def fn(line, task_name: str = None):
        doc = base(line, task_name)
        if doc is None or not doc.choices:
            return None
        gold_idx = doc.gold_index[0] if isinstance(doc.gold_index, list) else doc.gold_index
        return Doc(task_name=task_name, query=doc.query, choices=[doc.choices[gold_idx]], gold_index=0)

    return fn


def _gen_config(name_lang: str, language: Language, hf_subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"multi_wiki_qa:{name_lang}:gen",
        prompt_function=get_qa_prompt_function(language, _make_qa_adapter()),
        hf_repo="alexandrainst/multi-wiki-qa",
        hf_subset=hf_subset,
        evaluation_splits=["train"],
        hf_avail_splits=["train"],
        generation_size=100,
        stop_sequence=["\n"],
        metrics=[
            MultilingualQuasiExactMatchMetric(language, "prefix"),
            MultilingualQuasiF1ScoreMetric(language),
        ],
    )


def _bpb_config(name_lang: str, language: Language, hf_subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"multi_wiki_qa:{name_lang}:bpb",
        prompt_function=_make_bpb_prompt(language),
        hf_repo="alexandrainst/multi-wiki-qa",
        hf_subset=hf_subset,
        evaluation_splits=["train"],
        hf_avail_splits=["train"],
        generation_size=-1,
        stop_sequence=["\n"],
        metrics=[Metrics.target_bits_per_byte],
    )


# Languages with full QA template support and a matching subset via
# standardize_tag(lang.value). English is intentionally INCLUDED.
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

# Dataset subset codes that differ from standardize_tag(language.value).
# (e.g. Tagalog: standardize_tag("tgl") -> "fil", but the dataset uses "tl".)
_SUBSET_OVERRIDES = {
    Language.TAGALOG: "tl",
}


def _subset_for(language: Language) -> str:
    return _SUBSET_OVERRIDES.get(language, standardize_tag(language.value))


TASKS_TABLE = []
for language in _STANDARD_LANGUAGES:
    _subset = _subset_for(language)
    TASKS_TABLE.append(_gen_config(language.value, language, _subset))
    TASKS_TABLE.append(_bpb_config(language.value, language, _subset))

# Chinese: dataset splits Simplified (zh-cn) and Traditional (zh-tw) Mandarin.
for _zh_subset in ["zh-cn", "zh-tw"]:
    _zh_name = f"{Language.CHINESE.value}_{_zh_subset.split('-')[1]}"
    TASKS_TABLE.append(_gen_config(_zh_name, Language.CHINESE, _zh_subset))
    TASKS_TABLE.append(_bpb_config(_zh_name, Language.CHINESE, _zh_subset))

# Portuguese: dataset splits European (pt-pt) and Brazilian (pt-br) Portuguese.
for _pt_subset in ["pt-pt", "pt-br"]:
    _pt_name = f"{Language.PORTUGUESE.value}_{_pt_subset.split('-')[1]}"
    TASKS_TABLE.append(_gen_config(_pt_name, Language.PORTUGUESE, _pt_subset))
    TASKS_TABLE.append(_bpb_config(_pt_name, Language.PORTUGUESE, _pt_subset))
