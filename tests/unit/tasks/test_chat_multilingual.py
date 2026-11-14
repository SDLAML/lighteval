"""Tests of the multilingual chat evals (tasks/tasks/chat_multilingual.py): MMMLU (14 locales), INCLUDE (44 languages), chat prompt.

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_multilingual.py`. The subset file and the task table are checked
for shape and consistency; no dataset is read here (the rows of the subsets are checked against the drawn ids only in the draw itself).
The scorers are the chat metrics of chat_evals / chat_metrics.
"""

import json
import unittest

from lighteval.metrics.chat_metrics import chat_choice_metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_multilingual as cmu
from lighteval.tasks.tasks.chat_evals import mc_query


def _resp(*texts):
    return ModelResponse(text=list(texts), text_post_processed=list(texts))


def _doc(n_options: int = 4, gold: int = 2) -> Doc:
    letters = "ABCDEFGHIJ"[:n_options]
    return Doc(query="q", choices=list(letters), gold_index=gold, task_name="t")


def _score(doc, text):
    return chat_choice_metric(1).sample_level_fn.compute(doc, _resp(text))


MMMLU_LINE = {"Unnamed: 0": 7, "Question": "  Quelle est la capitale de la France ?  ", "A": "Lyon", "B": "Marseille", "C": "Paris",
              "D": "Nice", "Answer": "C", "Subject": "geography"}
INCLUDE_LINE = {"language": "Arabic", "country": "Jordan", "domain": "Business & Commerce", "subject": "Management",
                "regional_feature": "region implicit", "level": "Academic", "question": " \nSome question? ",
                "option_a": "one", "option_b": "two", "option_c": "three", "option_d": "four", "answer": "2"}


def test_mmmlu_prompt_is_the_mmlu_pro_chat_prompt():
    d = cmu.mmmlu_chat_prompt(MMMLU_LINE, "mmmlu_chat:fr")
    assert d.query == mc_query("Quelle est la capitale de la France ?", ["Lyon", "Marseille", "Paris", "Nice"])
    assert d.query.startswith("Answer the following multiple choice question.") and "'Answer: $LETTER'" in d.query
    assert d.choices == ["A", "B", "C", "D"] and d.gold_index == 2
    assert d.specific["subject"] == "geography" and d.specific["row"] == 7


def test_mmmlu_filter_keys_on_the_row_index():
    keep = cmu.mmmlu_filter(frozenset({7, 9}))
    assert keep({"Unnamed: 0": 7}) and keep({"Unnamed: 0": "9"}) and not keep({"Unnamed: 0": 8})


def test_include_prompt_and_gold():
    d = cmu.include_chat_prompt(INCLUDE_LINE, "include_chat:ara")
    assert d.query == mc_query("Some question?", ["one", "two", "three", "four"])
    assert d.choices == ["A", "B", "C", "D"] and d.gold_index == 2 and d.specific["country"] == "Jordan"


def test_include_row_key_is_stable_and_filters():
    key = cmu.include_row_key(INCLUDE_LINE)
    assert key == cmu.include_row_key(dict(INCLUDE_LINE)), "the same row gives the same id"
    assert cmu.include_row_key({**INCLUDE_LINE, "answer": "1"}) != key, "a changed answer is another row"
    assert cmu.include_filter(frozenset({key}))(INCLUDE_LINE) and not cmu.include_filter(frozenset())(INCLUDE_LINE)


def test_mmmlu_scoring_rules():
    doc = _doc(4, gold=2)                                             # gold C
    assert _score(doc, "<think>r</think>\n\nAnswer: C")["acc"] == 1.0
    assert _score(doc, "<think>r</think>\n\nAnswer: B")["acc"] == 0.0
    assert _score(doc, "")["acc"] == 0.0
    # the strict split (split_reasoning, upstream of the metric) leaves nothing after an unclosed block: the metric then gets ""
    unclosed = ModelResponse(text=["<think>the answer is C but I keep going"], text_post_processed=[""])
    assert chat_choice_metric(1).sample_level_fn.compute(doc, unclosed)["acc"] == 0.0, "an unclosed think block is no answer"
    assert _score(doc, "<think>x</think>\n\nAnswer: \\boxed{C}")["acc"] == 1.0


def test_include_scoring_uses_the_letter_of_the_options():
    doc = _doc(4, gold=0)                                             # gold A
    assert _score(doc, "<think>r</think>\n\nAnswer: A")["acc"] == 1.0
    assert _score(doc, "<think>r</think>\n\nAnswer: D")["acc"] == 0.0


def test_subset_file_shape_and_sizes():
    data = json.loads(cmu.SUBSETS_FILE.read_text())
    assert data["per_language"] == cmu.PER_LANG == 200
    assert set(data["mmmlu"]) == set(cmu.MMMLU_LOCALES) and len(data["mmmlu"]) == 14
    for locale, ids in data["mmmlu"].items():
        assert len(ids) == 200 and ids == sorted(ids) and len(set(ids)) == 200, locale
        assert all(0 <= i < 14042 for i in ids), locale                  # each locale has 14,042 test rows
    assert len(data["include"]) == 44
    for language, keys in data["include"].items():
        assert 0 < len(keys) <= 200 and keys == sorted(keys) and len(set(keys)) == len(keys), language
        assert all(len(k) == 32 for k in keys), language                 # md5 hex ids
    assert data["include"]["Uzbek"] and len(data["include"]["Uzbek"]) == 200


def test_task_table_names_and_pins():
    names = [t.name for t in cmu.TASKS_TABLE]
    assert len(names) == 58 and len(set(names)) == 58
    mmmlu = [t for t in cmu.TASKS_TABLE if t.name.startswith("mmmlu_chat:")]
    include = [t for t in cmu.TASKS_TABLE if t.name.startswith("include_chat:")]
    assert len(mmmlu) == 14 and len(include) == 44
    assert {t.hf_revision for t in mmmlu} == {cmu.MMMLU_REVISION} and {t.hf_revision for t in include} == {cmu._BASE_REVISION}
    assert all(t.generation_size == cmu.BUDGET and t.stop_sequence == [] for t in cmu.TASKS_TABLE)
    assert all(tuple(t.hf_avail_splits) == ("test",) and tuple(t.evaluation_splits) == ("test",) for t in cmu.TASKS_TABLE)
    assert "sqi" in {t.name.split(":")[1] for t in include} and "zho" in {t.name.split(":")[1] for t in include}


def test_subset_file_is_present():
    assert cmu.SUBSETS_FILE.exists(), "the subset file is part of the repo (draw on the login node)"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok  {name}")
            except unittest.SkipTest as e:
                print(f"skip {name}: {e}")
