# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from typing import Callable, Literal, Sequence

import numpy as np

from lighteval.metrics.metrics_sample import (
    ExactMatches,
    F1_score,
    LoglikelihoodAcc,
    NormalizedMultiChoiceProbability,
    Probability,
)
from lighteval.metrics.normalizations import (
    LogProbCharNorm,
    LogProbNormalization,
    LogProbTokenNorm,
    get_multilingual_normalizer,
)
from lighteval.metrics.utils.extractive_match_utils import (  # noqa: F401
    ExprExtractionConfig,
    ExtractionTarget,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.metrics.utils.metric_utils import SampleLevelComputation, SampleLevelMetric, SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.language import Language
from lighteval.utils.utils import as_list
from lighteval.utils.timeout import timeout


logger = logging.getLogger(__name__)


class LogLikelihoodAccMetric(SampleLevelMetric):
    def __init__(self, normalization: LogProbNormalization | None = None):
        """Creates an accuracy (loglikelihood) metric, which returns accuracy given normalization."""
        super().__init__(
            metric_name="acc" + (f"_{normalization.name}" if normalization else ""),
            sample_level_fn=LoglikelihoodAcc(logprob_normalization=normalization),
            category=SamplingMethod.LOGPROBS,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        )


class NormalizedMultiChoiceProbMetric(SampleLevelMetric):
    def __init__(
        self,
        normalization: LogProbNormalization | None = None,
        aggregation_function: Callable[[np.ndarray], float] = np.max,
    ):
        """Creates a normalized multi-choice probability metric, which returns the probability of the gold choice / sum of probabilities of all choices (after logprobs are normalized)."""
        super().__init__(
            metric_name="normalized_mc_prob" + (f"_{normalization.name}" if normalization else ""),
            sample_level_fn=NormalizedMultiChoiceProbability(
                log_prob_normalization=normalization, aggregation_function=aggregation_function
            ),
            category=SamplingMethod.LOGPROBS,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        )


class ProbabilityMetric(SampleLevelMetric):
    def __init__(
        self,
        normalization: LogProbTokenNorm | None = None,
        aggregation_function: Callable[[np.ndarray], float] = np.max,
    ):
        """Creates a probability metric, which returns the probability of the gold choice given normalization."""
        super().__init__(
            metric_name="prob" + (f"_{normalization.name}" if normalization else ""),
            sample_level_fn=Probability(normalization=normalization, aggregation_function=aggregation_function),
            category=SamplingMethod.LOGPROBS,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        )


class MultilingualQuasiF1ScoreMetric(SampleLevelMetric):
    def __init__(self, language: Language, aggregation_function: Callable[[list[float]], float] = max):
        """Creates a language-aware F1 score metric, which returns the F1 score.

        Args:
            language: The language of the samples.
            aggregation_function: Aggregation samples to use when multiple golds are present.
        """
        super().__init__(
            metric_name=f"f1_{language.value}",
            sample_level_fn=F1_score(
                normalize_gold=get_multilingual_normalizer(language),
                normalize_pred=get_multilingual_normalizer(language),
                aggregation_function=aggregation_function,
            ),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        )


class MultilingualQuasiExactMatchMetric(SampleLevelMetric):
    def __init__(
        self,
        language: Language,
        match_type: Literal["prefix", "suffix", "full"] = "full",
        aggregation_function: Callable[[list[float]], float] = max,
    ):
        """Creates a language-aware exact match metric, which returns the exact match score
        Args:
            language: The language of the samples.
            match_type: The type of match to use
                - "prefix": Prefixes must match
                - "suffix": Suffixes must match
                - "full": Full strings must match
            aggregation_function: Aggregation samples to use when multiple golds are present.
        """
        super().__init__(
            metric_name=f"exact_match_{language.value}_{match_type}",
            sample_level_fn=ExactMatches(
                normalize_gold=get_multilingual_normalizer(language),
                normalize_pred=get_multilingual_normalizer(language),
                aggregation_function=aggregation_function,
                type_exact_match=match_type,
            ),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        )


class MultilingualExtractiveMatchMetric(SampleLevelComputation):
    def __init__(
        self,
        language: Language = Language.ENGLISH,
        gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
        pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function: Callable[[list[float]], float] = max,
        fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
        extraction_mode: Literal["first_match", "any_match"] = "any_match",
        precision: int = 6,
        timeout_seconds: int = 5,
    ):
        """Creates a language-aware extractive match metric that extracts answers from the model's output.

        Known issues:
        - If the task is to simplify an expression, the metric might overestimate the accuracy. This is because if the model doesn't output any anchor for the extraction (e.g final answer is..),
            it's possible that the extracted prediction will be the expression to simplify. Because we do simplifications ourselves, it can thus happen that sympy will correctly simplify the expression,
            thus it will match gold, despite model not doing anything. PRs to fix this are welcome.

        - There is currently no StringExtractionConfig, so if the gold is \boxed{\text{Friday}} and model outputs Friday it will not match, because nothing will be extracted.

        Args:
            language: Language
                The language of the samples.
            gold_extraction_target: Sequence[ExtractionTarget]
                Extraction targets to use for gold answers. Defaults to extracting simple math expressions.
            pred_extraction_target: Sequence[ExtractionTarget]
                Extraction targets to use for predictions. Defaults to extracting simple math expressions.
            aggregation_function: Callable[[list[float]], float]
                Function to aggregate scores when multiple golds/predictions are present. Defaults to max.
            fallback_mode: Literal["no_fallback", "first_match"]
                How to perform extraction. Defaults to "first_match".
                - "no_fallback": Only use first successfully parsed matches
                - "first_match": Use the first successfully parsed match + first match irregardless the parsing success
            extraction_mode: Literal["first_match", "any_match"]
                - "first_match": Only tries to extract the first regex match if it fails no other matches are tried
                - "any_match": Tries to extract any regex match

            precision: int
                Number of decimal places to use when comparing numerical values. Defaults to 6.
            timeout_seconds: int
                Timeout for the extraction (each attempt) and comparison. Defaults to 5.

        """
        self.language = language
        self.gold_extraction_target = gold_extraction_target
        self.pred_extraction_target = pred_extraction_target
        self.aggregation_function = aggregation_function
        self.fallback_mode = fallback_mode
        self.extraction_mode = extraction_mode
        self.precision = precision
        self.timeout_seconds = timeout_seconds

    @timeout(2)
    def add_to_specifics_with_timeout(
        self, formatted_doc: Doc, extracted_predictions: list[list[str]], extracted_golds: list[list[str]]
    ) -> None:
        if formatted_doc.specific is None:
            formatted_doc.specific = {}

        formatted_doc.specific["extracted_predictions"] = [
            str(pred) for preds in extracted_predictions for pred in preds
        ]
        formatted_doc.specific["extracted_golds"] = [str(gold) for golds in extracted_golds for gold in golds]

    def compute(self, doc: Doc, model_response: ModelResponse) -> float:
        golds = doc.get_golds()
        predictions = model_response.final_text

        gold_extraction_regexes = get_extraction_regexes(doc, self.gold_extraction_target, self.language)
        pred_extraction_regexes = get_extraction_regexes(doc, self.pred_extraction_target, self.language)

        extracted_predictions = [
            extract_target_from_pred(
                pred, pred_extraction_regexes, self.fallback_mode, self.extraction_mode, self.timeout_seconds
            )
            for pred in predictions
        ]
        extracted_golds = [
            extract_target_from_pred(
                gold, gold_extraction_regexes, self.fallback_mode, self.extraction_mode, self.timeout_seconds
            )
            for gold in golds
        ]

        # Assert on empty gold and warn on empty pred
        if any(len(g) == 0 for g in extracted_golds):
            logger.warning(f"We did not manage to extract a gold in the correct format. Gold: {golds}")
            extracted_golds = [[gold] for gold in golds]

        if all(len(p) == 0 for p in extracted_predictions):
            logger.warning(
                f"We did not manage to extract a prediction in the correct format. Gold: {golds}, Pred: {predictions}"
            )

        # We have to use timeout because the sypmy to str conversion can be very slow
        try:
            self.add_to_specifics_with_timeout(doc, extracted_predictions, extracted_golds)
        except TimeoutError:  # noqa: E722
            logger.warning("Timeout when adding extracted predictions and golds to specific")

        return self.aggregation_function(
            [
                (
                    1.0
                    if any(
                        compare_gold_target(gold, pred, self.precision, timeout_seconds=self.timeout_seconds)
                        for gold in extracted_golds
                    )
                    else 0.0
                )
                for pred in extracted_predictions
            ]
        )


# ---------------------------------------------------------------------------
# MMLU per-category accuracy metric
# ---------------------------------------------------------------------------
import math as _math

_MMLU_STEM = frozenset([
    "abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine", "college_physics",
    "computer_security", "conceptual_physics", "electrical_engineering", "elementary_mathematics",
    "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_statistics", "machine_learning",
])
_MMLU_HUMANITIES = frozenset([
    "formal_logic", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
    "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law",
    "world_religions",
])
_MMLU_SOCIAL = frozenset([
    "econometrics", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
    "human_sexuality", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy",
])
# "Other" = any subject not in the above three sets

_MMLU_CATS = ("stem", "humanities", "social", "other")


def _subject_to_mmlu_cat(subject: str) -> str:
    s = subject.lower()
    if s in _MMLU_STEM:       return "stem"
    if s in _MMLU_HUMANITIES: return "humanities"
    if s in _MMLU_SOCIAL:     return "social"
    return "other"


def _mean_notnone(vals):
    """Mean of non-None values; returns nan if all are None."""
    filtered = [v for v in vals if v is not None]
    return float(np.mean(filtered)) if filtered else float("nan")


class _MMLUCategoryFn(SampleLevelComputation):
    """Computes acc, acc_norm (char), and optionally per-sample BPB, routed to
    the correct MMLU category (stem/humanities/social/other) via
    ``doc.specific["subject"]``."""

    def __init__(self, include_bpb: bool = False):
        self._acc_fn      = LoglikelihoodAcc(logprob_normalization=None)
        self._acc_norm_fn = LoglikelihoodAcc(logprob_normalization=LogProbCharNorm())
        self.include_bpb  = include_bpb

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict:
        acc      = self._acc_fn.compute(doc=doc, model_response=model_response)
        acc_norm = self._acc_norm_fn.compute(doc=doc, model_response=model_response)

        subject = (doc.specific or {}).get("subject", "")
        cat = _subject_to_mmlu_cat(subject)

        result: dict = {"acc": acc, "acc_norm": acc_norm}
        for c in _MMLU_CATS:
            result[f"acc_{c}"]      = acc      if c == cat else None
            result[f"acc_norm_{c}"] = acc_norm if c == cat else None

        if self.include_bpb:
            gold_ix      = as_list(doc.gold_index)[0]
            gold_logprob = model_response.logprobs[gold_ix]
            gold_bytes   = max(len(doc.choices[gold_ix].encode("utf-8")), 1)
            bpb = -gold_logprob / _math.log(2) / gold_bytes
            result["bpb"] = bpb
            for c in _MMLU_CATS:
                result[f"bpb_{c}"] = bpb if c == cat else None

        return result


def _make_mmlu_category_grouping(include_bpb: bool) -> SampleLevelMetricGrouping:
    keys_acc = (
        ["acc", "acc_norm"]
        + [f"acc_{c}" for c in _MMLU_CATS]
        + [f"acc_norm_{c}" for c in _MMLU_CATS]
    )
    keys_bpb = ["bpb"] + [f"bpb_{c}" for c in _MMLU_CATS] if include_bpb else []
    all_keys = keys_acc + keys_bpb

    corpus_fns: dict = {
        "acc":      np.mean,
        "acc_norm": np.mean,
        **{f"acc_{c}":      _mean_notnone for c in _MMLU_CATS},
        **{f"acc_norm_{c}": _mean_notnone for c in _MMLU_CATS},
    }
    if include_bpb:
        corpus_fns["bpb"] = np.mean
        corpus_fns.update({f"bpb_{c}": _mean_notnone for c in _MMLU_CATS})

    higher_is_better = {
        **{k: True  for k in keys_acc},
        **{k: False for k in keys_bpb},
    }

    return SampleLevelMetricGrouping(
        metric_name=all_keys,
        sample_level_fn=_MMLUCategoryFn(include_bpb=include_bpb),
        corpus_level_fn=corpus_fns,
        category=SamplingMethod.LOGPROBS,
        higher_is_better=higher_is_better,
    )


MMLUCategoryGroupingCF  = _make_mmlu_category_grouping(include_bpb=True)
"""CF variant: reports acc, acc_norm, bpb — overall and per MMLU category (stem/humanities/social/other)."""

MMLUCategoryGroupingMCF = _make_mmlu_category_grouping(include_bpb=False)
"""MCF variant: reports acc, acc_norm — overall and per MMLU category (no BPB for label-token scoring)."""
