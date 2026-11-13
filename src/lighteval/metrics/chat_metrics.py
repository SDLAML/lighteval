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

"""Metrics of the titan_eval chat evals (post-trained models: chat template, reasoning block, several samples per prompt).

One metric object scores ALL samples of a prompt (as many as the model returned: the task's own n, or the run's `num_samples`):
  <primary>   mean over the samples of the strict score (math: a `\\boxed{}` answer is required; multiple choice: an anchored
              "Answer: X" is required, `anchored_letter`)
  <second>    the lenient score of the same samples (math: the permissive extraction of `math_500`, an unboxed last expression
              counts; multiple choice: a stray letter anywhere counts), a DIAGNOSTIC number, never the headline
  pass@k      the unbiased estimator of Chen et al. 2021 over the primary scores, for k = 2, 4, 8, 16, 32 with k <= n
              (pass@1 is the primary itself); keys with k > n are simply not returned
The text scored is `model_response.final_text`, i.e. what follows the LAST `</think>` (an unclosed block gives "" under
`--strict-reasoning`); an empty text scores 0 without running the extractors. The samples are scored one by one through fresh
`ModelResponse(text=[sample])` objects (never `model_response[i]`, which drops `text_post_processed` and would score the reasoning).
"""

import logging
import re
from string import ascii_uppercase
from typing import Callable, Sequence

import numpy as np

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics_sample import SampleLevelComputation, SamplingMetric
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
)
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.language import Language


logger = logging.getLogger(__name__)

PASS_KS = (2, 4, 8, 16, 32)


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. 2021: 1 - C(n - c, k) / C(n, k) (n samples, c of them correct)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _math_extractor() -> MultilingualExtractiveMatchMetric:
    """The scorer of math_500 / math:gen (Metrics.expr_gold_metric): boxed answer first, then expressions; sympy / latex equivalence, precision 5."""
    return MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        fallback_mode="first_match",
        precision=5,
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        aggregation_function=max,
    )


def _letters_extractor() -> MultilingualExtractiveMatchMetric:
    """The DIAGNOSTIC multiple-choice number: any letter at all counts (the upstream gpqa_instruct_metric, try_extract_without_anchor=True). Gold = the letter itself."""
    return MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)],
        pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)],
        precision=6,
    )


# The anchored answer: the word "answer" (any case) followed by a colon and/or "is" (at least one of the two), optional markdown / brackets / "$" / \boxed{} \text{} around ONE
# capital letter A-J. Matches "Answer: C", "**Answer:** C", "Answer: $C$", "Answer: \boxed{C}", "Answer:C", "Answer: (C) text", "The answer is (C).", "Therefore, the answer
# is C", "Final answer: C"; not a bare letter, not "Option C is correct", not "answer: c" (the prompt asks for the capital letter), not "the answer I gave" (no colon / "is":
# "I" would be option I of a 10-option question). The LAST match of the text is the model's answer. A HEDGE is no answer: a second letter joined by or / and / "/" / "," / "&"
# right after the first ("Answer: B or C", "Answer: C, A", "\boxed{B \text{ or } C}") gives None, so naming the gold letter first never earns credit
# (agents/reward_checks/multiple_choice.md). (The lighteval anchored extractor misses "$C$", "\boxed{C}", "Answer:C" and "Answer: C) text" -- probed 11-13 -- and credits a
# later line that starts with a letter.)
_ANSWER = re.compile(
    r"(?i:\banswer\b)[ \t]*(?:\*\*)?[ \t]*(?:(?i:is)\b[ \t]*[:\uff1a]?|[:\uff1a])[ \t]*(?:\*\*)?[ \t]*[\(\[]?[ \t]*\$?[ \t]*"
    r"(?:\\(?:boxed|text|textbf|mathrm)\{)?[ \t]*[\(\[]?([A-J])(?![A-Za-z0-9])"
)
_HEDGE = re.compile(r"[ \t]*[\)\]\}\$\*]*[ \t]*(?:/|,|&|(?:\\text\{[ \t]*)?\b(?:or|and)\b(?:[ \t]*\})?)[ \t]*[\(\[]?\$?[ \t]*[A-J](?![A-Za-z0-9])")


def anchored_letter(pred: str, n_options: int) -> str | None:
    """The letter of the LAST anchored answer in `pred` ("Answer: X" and its variants, see _ANSWER), None when there is none, when it is a hedge (a second letter
    follows it, see _HEDGE) or when it is not one of the n options."""
    last = None
    for m in _ANSWER.finditer(pred):
        last = None if _HEDGE.match(pred, m.end()) else m.group(1)
    if last is None or ascii_uppercase.index(last) >= n_options:
        return None
    return last


def _math_scores(extractor: MultilingualExtractiveMatchMetric) -> Callable[[Doc, str], dict[str, float]]:
    """One extraction per sample: lenient = the permissive score; strict = the same score when the text holds a \\boxed (the box has priority 0 in
    the extractor, so it is what gets compared), else 0 (exactly BoxedOnlyExtractiveMatchMetric, without its `model_response[i]` slicing)."""

    def score(doc: Doc, pred: str) -> dict[str, float]:
        if not pred or not pred.strip():
            return {"avg@n": 0.0, "avg@n_lenient": 0.0}
        lenient = float(extractor.compute(doc, ModelResponse(text=[pred])))
        return {"avg@n": lenient if "\\boxed" in pred else 0.0, "avg@n_lenient": lenient}

    return score


def _choice_scores(unanchored: MultilingualExtractiveMatchMetric) -> Callable[[Doc, str], dict[str, float]]:
    def score(doc: Doc, pred: str) -> dict[str, float]:
        if not pred or not pred.strip():
            return {"acc": 0.0, "acc_unanchored": 0.0}
        gold = doc.choices[doc.gold_index if isinstance(doc.gold_index, int) else doc.gold_index[0]]
        a = 1.0 if anchored_letter(pred, len(doc.choices)) == gold else 0.0
        u = 1.0 if a == 1.0 else float(unanchored.compute(doc, ModelResponse(text=[pred])))     # an anchored hit is also an any-letter hit
        return {"acc": a, "acc_unanchored": u}

    return score


class ChatSamples(SamplingMetric, SampleLevelComputation):
    """Scores every sample of a prompt with `score(doc, text) -> {name: 0 or 1}` (names = `names`, the first one is the primary); see the module
    docstring. `n` is only the task's DEFAULT number of samples (what lighteval asks the model for unless the run sets `num_samples`)."""

    def __init__(self, n: int, score: Callable[[Doc, str], dict[str, float]], names: Sequence[str], ks: Sequence[int] = PASS_KS):
        super().__init__()
        self.n = n
        self.score = score
        self.names = tuple(names)
        self.primary = self.names[0]
        self.ks = tuple(ks)

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict[str, float]:
        if len(doc.get_golds()) != 1:
            raise ValueError("ChatSamples needs exactly one gold")
        preds = list(model_response.final_text)
        if not preds:
            raise ValueError("no completion in the model response")
        per = [self.score(doc, p) for p in preds]
        out: dict[str, float] = {name: float(np.mean([s[name] for s in per])) for name in self.names}
        n, c = len(preds), int(sum(1 for s in per if s[self.primary] == 1.0))
        for k in self.ks:
            if k <= n:
                out[f"pass@{k}"] = unbiased_pass_at_k(n, c, k)
        return out

    def num_samples(self) -> int:
        return self.n


def _grouping(n: int, score: Callable[[Doc, str], dict[str, float]], names: Sequence[str]) -> SampleLevelMetricGrouping:
    all_names = list(names) + [f"pass@{k}" for k in PASS_KS]
    return SampleLevelMetricGrouping(
        metric_name=all_names,
        sample_level_fn=ChatSamples(n, score, names),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={name: np.mean for name in all_names},
        higher_is_better={name: True for name in all_names},
    )


def chat_math_metric(n: int) -> SampleLevelMetricGrouping:
    """avg@n (strict: needs a \\boxed answer), avg@n_lenient (the permissive extraction), pass@k over the strict scores. n = default samples."""
    return _grouping(n, _math_scores(_math_extractor()), ("avg@n", "avg@n_lenient"))


def chat_choice_metric(n: int) -> SampleLevelMetricGrouping:
    """acc (anchored letter), acc_unanchored (any letter: diagnostic), pass@k over the anchored scores. n = default samples."""
    return _grouping(n, _choice_scores(_letters_extractor()), ("acc", "acc_unanchored"))
