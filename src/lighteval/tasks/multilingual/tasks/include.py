"""
name:
INCLUDE

dataset:
CohereLabs/include-base-44 (the full benchmark) and CohereLabs/include-lite-44 (a smaller test set of the same 44 languages)

abstract:
INCLUDE (Romanou et al., 2024) is a regional-knowledge multiple-choice benchmark in 44 languages: exam questions
(academic, professional, license exams) collected in the countries where the languages are spoken, 4 options, a part of
them region-explicit, a part region-agnostic. include-base-44: 22,639 test questions + 890 validation questions (the
few-shot pool); include-lite-44: 10,770 test questions (a subset of base's test questions, verbatim; 245 of them with other choices / answer than in base), no
validation split, `choices` list instead of option_a .. option_d. License Apache-2.0.

Letter format `mcf_em` (the format of mmlu:mcf_em / piqa:mcf_em): "Question: ... A. ... B. ... C. ... D. ... Answer:", the
model writes the letter, exact match. The labels are English for every language (as mmlu_prox does; no dependence on
translation literals, which the fork lacks for some of the 44 languages). Shots NEVER come from the evaluated split:
`include:*` takes them from the language's validation split of include-base-44; `include_lite:*` (no validation split of its
own) takes the first K validation items of include-base-44 of the same language (`few_shots_rows`).

languages:
44, named here by ISO 639-3 (the dataset configs are English names: Albanian, Arabic, ...)

tags:
multilingual, knowledge, multiple-choice, regional

paper:
https://arxiv.org/abs/2411.19799
"""

from functools import lru_cache
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Pinned dataset revisions (the snapshots in the offline cache, downloaded 2026-11-13).
_BASE_REPO, _BASE_REVISION = "CohereLabs/include-base-44", "d2e1f6015f67a43c02a9a68db98e2298e2d6a660"
_LITE_REPO, _LITE_REVISION = "CohereLabs/include-lite-44", "a027ccc923428c0c43c295a91efcc39a1e47fa60"

# dataset config (English language name) -> ISO 639-3 code used in the task name
_LANGUAGES = {
    "Albanian": "sqi",
    "Arabic": "ara",
    "Armenian": "hye",
    "Azerbaijani": "aze",
    "Basque": "eus",
    "Belarusian": "bel",
    "Bengali": "ben",
    "Bulgarian": "bul",
    "Chinese": "zho",
    "Croatian": "hrv",
    "Dutch": "nld",
    "Estonian": "est",
    "Finnish": "fin",
    "French": "fra",
    "Georgian": "kat",
    "German": "deu",
    "Greek": "ell",
    "Hebrew": "heb",
    "Hindi": "hin",
    "Hungarian": "hun",
    "Indonesian": "ind",
    "Italian": "ita",
    "Japanese": "jpn",
    "Kazakh": "kaz",
    "Korean": "kor",
    "Lithuanian": "lit",
    "Malay": "msa",
    "Malayalam": "mal",
    "Nepali": "nep",
    "North Macedonian": "mkd",
    "Persian": "fas",
    "Polish": "pol",
    "Portuguese": "por",
    "Russian": "rus",
    "Serbian": "srp",
    "Spanish": "spa",
    "Tagalog": "tgl",
    "Tamil": "tam",
    "Telugu": "tel",
    "Turkish": "tur",
    "Ukrainian": "ukr",
    "Urdu": "urd",
    "Uzbek": "uzb",
    "Vietnamese": "vie",
}


def _choices(line) -> list[str]:
    """include-base-44 rows hold option_a .. option_d; include-lite-44 rows hold a `choices` list (the same questions, other schema)."""
    if line.get("choices") is not None:
        return [str(c) for c in line["choices"]]
    return [line["option_a"], line["option_b"], line["option_c"], line["option_d"]]


def include_mcf_prompt(line, task_name: str = None):
    choices = _choices(line)
    options = "\n".join(f" {letter}. {choice}" for letter, choice in zip(ascii_uppercase, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {letter}" for letter in ascii_uppercase[:4]],
        gold_index=int(line["answer"]),
        specific={"subject": line.get("subject"), "domain": line.get("domain"), "country": line.get("country"),
                  "level": line.get("level"), "regional_feature": line.get("regional_feature")},
    )


def _base_validation_rows(language: str):
    """The validation split of include-base-44 for one language, as a `few_shots_rows` callable (the lite set has no dev split)."""

    @lru_cache(maxsize=1)
    def rows() -> list[dict]:
        from lighteval.tasks.lighteval_task import load_dataset_pinned

        return [dict(r) for r in load_dataset_pinned(_BASE_REPO, language, split="validation", revision=_BASE_REVISION)]

    return rows


_COMMON = dict(
    prompt_function=include_mcf_prompt,
    evaluation_splits=["test"],
    generation_size=1,
    stop_sequence=["\n"],
    metrics=[Metrics.exact_match],
)

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"include:{code}:mcf_em",
        hf_repo=_BASE_REPO,
        hf_revision=_BASE_REVISION,
        hf_subset=language,
        hf_avail_splits=["test", "validation"],
        few_shots_split="validation",
        **_COMMON,
    )
    for language, code in _LANGUAGES.items()
] + [
    LightevalTaskConfig(
        name=f"include_lite:{code}:mcf_em",
        hf_repo=_LITE_REPO,
        hf_revision=_LITE_REVISION,
        hf_subset=language,
        hf_avail_splits=["test"],
        few_shots_rows=_base_validation_rows(language),
        few_shots_select="sequential",
        **_COMMON,
    )
    for language, code in _LANGUAGES.items()
]
