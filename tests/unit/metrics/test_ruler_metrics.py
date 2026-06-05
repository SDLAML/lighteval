# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from lighteval.metrics.metrics_sample import RulerStringMatch, RulerStringMatchAny
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


def make_doc(golds: list[str], task_name: str) -> Doc:
    return Doc(query="", choices=golds, gold_index=list(range(len(golds))), task_name=task_name)


@pytest.mark.parametrize(
    ("prediction", "golds"),
    [
        ("eiffel tower", ["The Eiffel Tower"]),
        ("Answer: eiffel tower", ["The Eiffel Tower"]),
        ("ANSWER: the eiffel tower", ["eiffel tower"]),
        ("the answer is london", ["Paris", "London"]),
    ],
)
def test_ruler_qa_match_uses_normalized_substring_semantics(prediction, golds):
    metric = RulerStringMatch()
    doc = make_doc(golds, "ruler_512:qa_2|0")

    assert metric.compute(doc, ModelResponse(text=[prediction])) == 1.0


def test_ruler_qa_parses_answer_prefix():
    assert RulerStringMatchAny._parse_output("Answer: Paris\nExtra") == "Paris"


def test_ruler_qa_returns_zero_when_no_gold_matches():
    metric = RulerStringMatch()
    doc = make_doc(["Paris"], "ruler_512:qa_1|0")

    assert metric.compute(doc, ModelResponse(text=["London"])) == 0.0


def test_ruler_non_qa_full_recall_scores_one():
    metric = RulerStringMatch()
    doc = make_doc(["alpha", "beta"], "ruler_512:cwe|0")

    assert metric.compute(doc, ModelResponse(text=["alpha beta gamma"])) == 1.0


def test_ruler_non_qa_partial_recall_scores_fraction():
    metric = RulerStringMatch()
    doc = make_doc(["alpha", "beta", "gamma"], "ruler_512:vt|0")

    assert metric.compute(doc, ModelResponse(text=["alpha gamma"])) == pytest.approx(2 / 3)


def test_ruler_non_qa_empty_prediction_scores_zero():
    metric = RulerStringMatch()
    doc = make_doc(["alpha", "beta"], "ruler_512:fwe|0")

    assert metric.compute(doc, ModelResponse(text=[""])) == 0.0
