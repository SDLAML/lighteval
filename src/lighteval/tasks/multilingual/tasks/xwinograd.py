"""
name:
Xwinograd

dataset:
Muennighoff/xwinograd

abstract:
Xwinograd multilingual benchmark.

languages:
chinese, english, french, japanese, portuguese, russian

tags:
multilingual, multiple-choice, reasoning

paper:
"""

import random
from functools import lru_cache, partial

from langcodes import standardize_tag

from lighteval.metrics.dynamic_metrics import (
    LogLikelihoodAccMetric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbTokenNorm
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    winogrand_adapter,
)
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.tasks.tasks.winogrande import winogrande_mcf_prompt
from lighteval.utils.language import Language


# Pinned dataset revisions (the snapshots in the offline cache): XWinograd itself, and English WinoGrande for the demonstrations.
_REVISION = "90b619ef5278605cd4f572e3e8506ab16afd44f2"
_WINOGRANDE_REPO = "allenai/winogrande"
_WINOGRANDE_REVISION = "01e74176c63542e6b0bcb004dcdea22d94fb67b5"

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"xwinograd_{language.value}_{formulation.name.lower()}",
        prompt_function=get_continuation_prompt_function(
            language, partial(winogrand_adapter, language), formulation=formulation
        ),
        hf_repo="Muennighoff/xwinograd",
        hf_revision=_REVISION,
        hf_subset=standardize_tag(language.value) if language != Language.JAPANESE else "jp",
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        metrics=[
            LogLikelihoodAccMetric(normalization=None),
            LogLikelihoodAccMetric(normalization=LogProbTokenNorm()),
            LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
        ],
    )
    for language in [
        Language.ENGLISH,
        Language.FRENCH,
        Language.JAPANESE,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.CHINESE,
    ]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]


# ---- letter format (11-13, multilingual suite): the prompt and the scoring of `winogrande:mcf_em` ---------------------------------------
# "Fill in the blank: <sentence with ___>\n A. <option1>\n B. <option2>\nAnswer:" (English words for every language, as winogrande.py),
# the model writes A or B, exact match. XWinograd has ONLY a test split, so the shots are 8 fixed items of the English WinoGrande TRAIN
# split (same row format, seeded draw, gold letters alternating), never an item of the evaluated language.


@lru_cache(maxsize=1)
def _english_winogrande_demos(n: int = 8) -> list[dict]:
    from datasets import load_dataset

    train = load_dataset(_WINOGRANDE_REPO, "winogrande_xl", split="train", revision=_WINOGRANDE_REVISION)
    rng = random.Random(1234)
    by_answer = {"1": [], "2": []}
    for i in rng.sample(range(len(train)), 200):
        by_answer[str(train[i]["answer"])].append(train[i])
    rows = []
    for k in range(n):
        r = by_answer["2" if k % 2 == 0 else "1"].pop(0)
        rows.append({"sentence": r["sentence"], "option1": r["option1"], "option2": r["option2"], "answer": str(r["answer"])})
    return rows


TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"xwinograd:{language.value}:mcf_em",
        prompt_function=winogrande_mcf_prompt,
        hf_repo="Muennighoff/xwinograd",
        hf_revision=_REVISION,
        hf_subset=standardize_tag(language.value) if language != Language.JAPANESE else "jp",
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        few_shots_rows=_english_winogrande_demos,
        few_shots_select="sequential",
        generation_size=1,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
    )
    for language in [
        Language.ENGLISH,
        Language.FRENCH,
        Language.JAPANESE,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.CHINESE,
    ]
]
