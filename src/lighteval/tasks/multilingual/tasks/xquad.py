"""
name:
Xquad

dataset:
google/xquad

abstract:
Reading Comprehension (RC) tasks evaluate a model's ability to understand and
extract information from text passages. These tasks typically involve answering
questions based on given contexts, spanning multiple languages and formats. Add
RC tasks supporting about 130 unique languages/scripts. SQuAD - like XQuAD:
Cross-lingual Question Answering Dataset, extending SQuAD to 11 languages.

languages:
arabic, chinese, english, german, greek, hindi, romanian, russian, spanish,
thai, turkish, vietnamese

tags:
multilingual, qa

paper:
https://arxiv.org/abs/1910.11856
"""

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import MultilingualSquadOfficialMetric
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.qa import get_qa_prompt_function
from lighteval.utils.language import Language


# Pinned dataset revision (google/xquad): the snapshot in the offline cache; `load_dataset(revision=...)` reads exactly it.
_REVISION = "51adfef1c1287aab1d2d91b5bead9bcfb9c68583"


_LANGUAGES = [
    Language.ARABIC,
    Language.GERMAN,
    Language.GREEK,
    Language.ENGLISH,
    Language.SPANISH,
    Language.HINDI,
    Language.ROMANIAN,
    Language.RUSSIAN,
    Language.THAI,
    Language.TURKISH,
    Language.VIETNAMESE,
    Language.CHINESE,
]


def _adapter(line):
    return {
        "question": line["question"],
        "context": line["context"],
        "choices": [ans for ans in line["answers"]["text"] if len(ans) > 0],
    }


def _bpb_prompt(language: Language):
    """Reuse the gen QA query, but score only the first gold continuation (BPB)."""
    base = get_qa_prompt_function(language, _adapter)

    def fn(line, task_name: str = None):
        doc = base(line, task_name)
        if doc is None or not doc.choices:
            return None
        gold_idx = doc.gold_index[0] if isinstance(doc.gold_index, list) else doc.gold_index
        return Doc(task_name=task_name, query=doc.query, choices=[doc.choices[gold_idx]], gold_index=0)

    return fn


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"xquad:{language.value}:gen",
        prompt_function=get_qa_prompt_function(language, _adapter),
        hf_repo="google/xquad",
        hf_revision=_REVISION,
        hf_subset=f"xquad.{standardize_tag(language.value)}",
        evaluation_splits=("validation",),
        few_shots_split="validation",
        generation_size=400,
        stop_sequence=["\n",],
        # 11-13: the official SQuAD arithmetic (equality / Counter F1, max over the golds) with the language's own normaliser;
        # metric names "em" and "f1" for every language (the Quasi metrics put the language in the name and use a lenient
        # prefix match / set-based F1). See MultilingualSquadOfficialMetric.
        metrics=(
            MultilingualSquadOfficialMetric(language, "em"),
            MultilingualSquadOfficialMetric(language, "f1"),
        ),
        # 1 (11-13 review fix): the normaliser removes the Unicode bidi marks (every language; xquad ara em +2.3 points). Prompts and
        # generations are those of version 0.
        version=1,
    )
    for language in _LANGUAGES
]

# Decoupled BPB: same query, score only the gold continuation (language-agnostic).
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"xquad:{language.value}:bpb",
        prompt_function=_bpb_prompt(language),
        hf_repo="google/xquad",
        hf_revision=_REVISION,
        hf_subset=f"xquad.{standardize_tag(language.value)}",
        evaluation_splits=("validation",),
        few_shots_split="validation",
        generation_size=-1,
        stop_sequence=["\n",],
        metrics=[Metrics.target_bits_per_byte],
    )
    for language in _LANGUAGES
]
