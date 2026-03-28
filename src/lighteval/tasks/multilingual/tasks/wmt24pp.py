"""
name:
WMT24++

dataset:
google/wmt24pp

abstract:
WMT24++ is an extension of the WMT2024 general translation shared task test
sets, covering 55+ language pairs with English as one side. Data includes
post-edited professional translations and quality flags.

Tasks are 0-shot generation with chrF++ and BLEU-4 metrics.
The public selector is `wmt24pp:{lp}|0`, which expands to the internal
directional tasks `wmt24pp:en_to_x:{lp}` and `wmt24pp:x_to_en:{lp}`.

NOTE: The dataset's "train" split is the evaluation data (WMT test sets are
published in HuggingFace with split="train").
Few-shot evaluation reuses this same split, since no separate few-shot split is
published for WMT24++ in the dataset.

NOTE: `lp` column values use regional codes (e.g., "de_DE", "fr_FR"). Tasks
are named using these codes directly. Check the dataset for all available lp
values: https://huggingface.co/datasets/google/wmt24pp

tags:
multilingual, translation

paper:
https://arxiv.org/abs/2412.06378
"""

from functools import partial

from lighteval.tasks.multilingual.utils.translation import (
    TRANSLATION_METRICS,
    EnglishCentricTranslationConfig,
    build_english_centric_translation_tasks,
)
from lighteval.utils.language import Language


_TRANSLATION_METRICS = TRANSLATION_METRICS

_LANGUAGE_CONFIGS = [
    ("de_DE", Language.GERMAN),
    ("fr_FR", Language.FRENCH),
    ("cs_CZ", Language.CZECH),
    ("es_MX", Language.SPANISH),
    ("ru_RU", Language.RUSSIAN),
    ("zh_CN", Language.CHINESE),
    ("ja_JP", Language.JAPANESE),
    ("uk_UA", Language.UKRAINIAN),
    ("hi_IN", Language.HINDI),
    ("ar_EG", Language.ARABIC),
    ("ko_KR", Language.KOREAN),
    ("pt_BR", Language.PORTUGUESE),
    ("tr_TR", Language.TURKISH),
    ("pl_PL", Language.POLISH),
    ("he_IL", Language.HEBREW),
    ("nl_NL", Language.DUTCH),
    ("it_IT", Language.ITALIAN),
    ("sv_SE", Language.SWEDISH),
    ("fi_FI", Language.FINNISH),
    ("vi_VN", Language.VIETNAMESE),
    ("bn_IN", Language.BENGALI),
    ("th_TH", Language.THAI),
    ("id_ID", Language.INDONESIAN),
    ("hu_HU", Language.HUNGARIAN),
]


def _make_forward_adapter(_lp_code):
    """Adapter for en→X direction: source=English, target=other language."""
    return lambda line: {
        "source_text": line["source"],
        "target_text": line["target"],
    }


def _make_reverse_adapter(_lp_code):
    """Adapter for X→en direction: source=other language, target=English."""
    return lambda line: {
        "source_text": line["target"],
        "target_text": line["source"],
    }


def _make_filter(lp_code):
    """Skip bad source sentences inside a specific WMT24++ pair config."""
    return partial(lambda _lp, x: not x.get("is_bad_source", False), lp_code)


TASKS_TABLE = build_english_centric_translation_tasks(
    base_name="wmt24pp",
    hf_repo="google/wmt24pp",
    language_configs=[
        EnglishCentricTranslationConfig(
            task_id=lp_code,
            target_language=target_lang,
            forward_hf_subset=f"en-{lp_code}",
            reverse_hf_subset=f"en-{lp_code}",
            forward_adapter=_make_forward_adapter(lp_code),
            reverse_adapter=_make_reverse_adapter(lp_code),
            hf_filter=_make_filter(lp_code),
            hf_avail_splits=("train",),
            evaluation_splits=("train",),
            few_shots_split="train",
            few_shots_select="random_sampling",
        )
        for lp_code, target_lang in _LANGUAGE_CONFIGS
    ],
    metrics=_TRANSLATION_METRICS,
)
