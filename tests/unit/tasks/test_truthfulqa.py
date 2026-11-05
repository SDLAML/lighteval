import pytest

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks.truthfulqa import truthful_qa_generative_prompt, truthfulqa_gen


@pytest.fixture
def doc():
    return truthful_qa_generative_prompt(
        {
            "question": " Which city? ",
            "correct_answers": [" New York City ", " NYC. ", "", "   "],
            "incorrect_answers": [" London ", " Paris. ", "", "   "],
        },
        task_name="truthfulqa:gen",
    )


def test_generative_references_preserve_answers_and_exclude_incorrect_golds(doc):
    assert doc.query == "Which city?"
    assert doc.choices == ["New York City.", "NYC.", "London.", "Paris."]
    assert doc.get_golds() == ["New York City.", "NYC."]


@pytest.mark.parametrize(
    ("prediction", "f1", "em"),
    [
        ("NEW YORK CITY!", 1.0, 1.0),
        ("nyc", 1.0, 1.0),
        ("New York", 0.8, 0.0),
        ("London", 0.0, 0.0),
        ("I have no comment", 0.0, 0.0),
        ("", 0.0, 0.0),
    ],
)
def test_generative_metrics_score_correct_references(doc, prediction, f1, em):
    response = ModelResponse(text=[prediction])
    scores = {
        metric.metric_name: metric.sample_level_fn.compute(doc, response) for metric in truthfulqa_gen.metrics
    }

    assert scores == pytest.approx({"f1": f1, "em": em})
