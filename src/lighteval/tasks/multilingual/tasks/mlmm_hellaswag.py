"""
name:
Mlmm Hellaswag

dataset:
alexandrainst/m_hellaswag (most languages)
jon-tow/okapi_hellaswag (Chinese)

abstract:
Hellaswag is a commonsense reasoning task that requires models to complete a
given scenario with the most plausible ending. It tests the model's ability to
understand and reason about everyday situations and human behavior.
MLMM-Hellaswag: Multilingual adaptation of Hellaswag.

Refactored to use unified :cf / :mcf / :mcf_em suffixes consistent with English hellaswag.py.
Few-shot (titan_eval, 11-16): :mcf_em takes up to 5 demonstrations from 5 rows of the language's own single split, held out and not scored
(see HELDOUT_SHOTS below); :cf and :mcf stay 0-shot only.
HellaSwag has 4 candidate continuations; the MCF formulation labels them A–D and scores (or
generates) the label token. English is excluded — use hellaswag.py for English evaluation.

languages:
arabic, armenian, basque, bengali, catalan, chinese, croatian, danish, dutch,
french, german, gujarati, hindi, hungarian, icelandic, indonesian, italian,
kannada, malayalam, marathi, nepali, portuguese, romanian, russian,
serbian, slovak, spanish, swedish, tamil, telugu, ukrainian, vietnamese

tags:
multilingual, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2306.07610
"""

from functools import lru_cache, partial

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.tasks.heldout_shots import content_key, load_rows, select_shots
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
from lighteval.utils.language import Language


_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]

# Languages covered by alexandrainst/m_hellaswag and jon-tow/okapi_hellaswag.
# English excluded — use hellaswag.py for English evaluation.
# Pinned dataset revisions: alexandrainst/m_hellaswag (every language but Chinese) and jon-tow/okapi_hellaswag (Chinese), the snapshots in the offline cache.
_REVISION = "9d31dc982bd6285e081e3e3136332a38b9c1d7b7"
_REVISION_ZHO = "fb6b5f2b48a6348ae2a9467a6714c3a054fa8d85"


_LANGUAGES = [
    Language.ARABIC,
    Language.BENGALI,
    Language.CATALAN,
    Language.DANISH,
    Language.GERMAN,
    Language.SPANISH,
    Language.BASQUE,
    Language.FRENCH,
    Language.GUJARATI,
    Language.HINDI,
    Language.CROATIAN,
    Language.HUNGARIAN,
    Language.ARMENIAN,
    Language.INDONESIAN,
    Language.ICELANDIC,
    Language.ITALIAN,
    Language.KANNADA,
    Language.MALAYALAM,
    Language.MARATHI,
    Language.NEPALI,
    Language.DUTCH,
    Language.PORTUGUESE,
    Language.ROMANIAN,
    Language.RUSSIAN,
    Language.SLOVAK,
    Language.SERBIAN,
    Language.SWEDISH,
    Language.TAMIL,
    Language.TELUGU,
    Language.UKRAINIAN,
    Language.VIETNAMESE,
    Language.CHINESE,
]


def _hellaswag_adapter(line):
    return {
        "activity_label": line.get("activity_label", ""),
        "ctx_a": line["ctx_a"],
        "ctx_b": line["ctx_b"],
        "continuations": line["endings"],
        "gold_idx": int(line["label"]),
    }


# Few-shot (owner 11-16, the base `mcf_em` variant only): the dataset has ONE split (`val` / `validation`), so a small FIXED slice of it is held out as the
# demonstrations of every prompt of the language and is NOT scored (the base_extra convention, heldout_shots.py). The rule is run on the pinned rows at
# load time (no constants): `select_shots` over the content key of every row, 5 shots, labels distinct first; every row with the content of a shot is
# left out of the scored items too (a duplicated row must not be its own demonstration). Tests: tests/unit/tasks/test_base_cot.py.
HELDOUT_SHOTS = 5


def _source(lang):
    if lang == Language.CHINESE:
        return "jon-tow/okapi_hellaswag", _REVISION_ZHO, "validation"
    return "alexandrainst/m_hellaswag", _REVISION, "val"


def _row_key(row) -> str:
    return content_key(row["activity_label"], row["ctx_a"], row["ctx_b"], *row["endings"])


@lru_cache(maxsize=None)
def _heldout(lang_value: str) -> tuple[tuple, frozenset]:
    """(the demonstration rows of a language in the order they are shown, the content keys that are not scored)."""
    lang = Language(lang_value)
    repo, revision, split = _source(lang)
    rows = load_rows(repo, standardize_tag(lang.value), split, revision)
    keys = [_row_key(r) for r in rows]
    chosen = select_shots(keys, labels=[str(r["label"]) for r in rows], k=HELDOUT_SHOTS)
    return tuple(rows[i] for i in chosen), frozenset(keys[i] for i in chosen)


def _shot_rows(lang) -> list[dict]:
    return [dict(r) for r in _heldout(lang.value)[0]]


def _scored_only(prompt_function, lang):
    """The prompt function of the language, returning None for a held-out row when it is NOT building a demonstration."""
    keys = None

    def wrapped(line, task_name: str = None):
        nonlocal keys
        if not line.get("__few_shots"):
            keys = keys if keys is not None else _heldout(lang.value)[1]
            if _row_key(line) in keys:
                return None
        return prompt_function(line, task_name)

    return wrapped


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"mlmm_hellaswag:{lang.value}:cf",
        prompt_function=get_hellaswag_prompt_function(
            language=lang,
            adapter=_hellaswag_adapter,
            formulation=CFFormulation(),
        ),
        hf_repo="alexandrainst/m_hellaswag" if lang != Language.CHINESE else "jon-tow/okapi_hellaswag",
        hf_revision=_REVISION_ZHO if lang == Language.CHINESE else _REVISION,
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["val" if lang != Language.CHINESE else "validation"],
        hf_avail_splits=["val" if lang != Language.CHINESE else "validation"],
        few_shots_split=None,
        metrics=_CF_METRICS,
    )
    for lang in _LANGUAGES
]

# MCF variant: labeled options in prompt, score label tokens via logprobs
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mlmm_hellaswag:{lang.value}:mcf",
        prompt_function=get_hellaswag_prompt_function(
            language=lang,
            adapter=_hellaswag_adapter,
            formulation=MCFFormulation(),
        ),
        hf_repo="alexandrainst/m_hellaswag" if lang != Language.CHINESE else "jon-tow/okapi_hellaswag",
        hf_revision=_REVISION_ZHO if lang == Language.CHINESE else _REVISION,
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["val" if lang != Language.CHINESE else "validation"],
        hf_avail_splits=["val" if lang != Language.CHINESE else "validation"],
        few_shots_split=None,
        metrics=_MCF_METRICS,
    )
    for lang in _LANGUAGES
]

# Greedy variant: MCF-style prompt, generate 1 token, exact match
TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mlmm_hellaswag:{lang.value}:mcf_em",
        prompt_function=_scored_only(
            get_hellaswag_prompt_function(
                language=lang,
                adapter=_hellaswag_adapter,
                formulation=MCFFormulation(),
            ),
            lang,
        ),
        hf_repo="alexandrainst/m_hellaswag" if lang != Language.CHINESE else "jon-tow/okapi_hellaswag",
        hf_revision=_REVISION_ZHO if lang == Language.CHINESE else _REVISION,
        hf_subset=standardize_tag(lang.value),
        evaluation_splits=["val" if lang != Language.CHINESE else "validation"],
        hf_avail_splits=["val" if lang != Language.CHINESE else "validation"],
        few_shots_rows=partial(_shot_rows, lang),
        few_shots_select="sequential",  # the first K of the 5 held-out demonstrations, in order (|0 still runs 0-shot, on the scored rows only)
        generation_size=1,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
        version=1,  # 1 (11-16): the 5 held-out demonstration rows of a language are not scored, at any K (0: all rows, 0-shot only)
    )
    for lang in _LANGUAGES
]
