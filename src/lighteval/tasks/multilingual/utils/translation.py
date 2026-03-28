from collections.abc import Callable, Sequence
from dataclasses import dataclass

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import Metric
from lighteval.tasks.lighteval_task import LightevalTaskConfig

# Canonical metric set for English-centric translation tasks.
# chrf++ and BLEU are instant (sacrebleu); COMET-22 requires unbabel-comet (pip install lighteval[multilingual]).
try:
    import comet  # noqa: F401

    _comet_available = True
except ImportError:
    _comet_available = False

TRANSLATION_METRICS: list[Metrics] = [
    Metrics.chrf_plus,
    Metrics.bleu,
    *([ Metrics.comet22] if _comet_available else []),
]
from lighteval.tasks.templates.translation import (
    TranslationInput,
    get_translation_prompt_function,
)
from lighteval.tasks.templates.utils.formulation import CFFormulation
from lighteval.utils.language import Language


@dataclass(frozen=True)
class EnglishCentricTranslationConfig:
    task_id: str
    target_language: Language
    forward_hf_subset: str
    reverse_hf_subset: str
    forward_adapter: Callable[[dict], TranslationInput | None]
    reverse_adapter: Callable[[dict], TranslationInput | None]
    hf_filter: Callable[[dict], bool] | None = None
    hf_avail_splits: Sequence[str] = ("train", "validation", "test")
    evaluation_splits: Sequence[str] = ("validation",)
    few_shots_split: str | None = None
    few_shots_select: str | None = None


def build_english_centric_translation_tasks(
    *,
    base_name: str,
    hf_repo: str,
    language_configs: Sequence[EnglishCentricTranslationConfig],
    metrics: Sequence[Metric | Metrics],
    generation_size: int = 300,
    stop_sequence: Sequence[str] = ("\n",),
    version: int = 0,
) -> list[LightevalTaskConfig]:
    tasks = []

    for config in language_configs:
        tasks.append(
            LightevalTaskConfig(
                name=f"{base_name}:en_to_x:{config.task_id}",
                prompt_function=get_translation_prompt_function(
                    source_language=Language.ENGLISH,
                    target_language=config.target_language,
                    adapter=config.forward_adapter,
                    formulation=CFFormulation(),
                ),
                hf_repo=hf_repo,
                hf_subset=config.forward_hf_subset,
                hf_filter=config.hf_filter,
                hf_avail_splits=config.hf_avail_splits,
                evaluation_splits=config.evaluation_splits,
                few_shots_split=config.few_shots_split,
                few_shots_select=config.few_shots_select,
                generation_size=generation_size,
                metrics=metrics,
                stop_sequence=stop_sequence,
                version=version,
            )
        )
        tasks.append(
            LightevalTaskConfig(
                name=f"{base_name}:x_to_en:{config.task_id}",
                prompt_function=get_translation_prompt_function(
                    source_language=config.target_language,
                    target_language=Language.ENGLISH,
                    adapter=config.reverse_adapter,
                    formulation=CFFormulation(),
                ),
                hf_repo=hf_repo,
                hf_subset=config.reverse_hf_subset,
                hf_filter=config.hf_filter,
                hf_avail_splits=config.hf_avail_splits,
                evaluation_splits=config.evaluation_splits,
                few_shots_split=config.few_shots_split,
                few_shots_select=config.few_shots_select,
                generation_size=generation_size,
                metrics=metrics,
                stop_sequence=stop_sequence,
                version=version,
            )
        )

    return tasks
