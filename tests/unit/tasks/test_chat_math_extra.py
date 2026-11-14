"""Chat math extras (tasks/tasks/chat_math_extra.py): fusion_aya_math_chat (per language), gsm8k_chat, olympiadbench_chat, hmmt_feb_2025_chat.

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_math_extra.py`. No GPU, no dataset read: the tests use the task
definitions and the metric only.
"""

from lighteval.metrics.chat_metrics import chat_math_metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.multilingual.tasks.fusion_aya_math import fusion_aya_gold
from lighteval.tasks.tasks import chat_math_extra as cme


def _cfg(name):
    return next(c for c in cme.TASKS_TABLE if c.name == name)


def _score(task_name, line, text):
    doc = _cfg(task_name).prompt_function(line, "t")
    resp = ModelResponse(text=[text], text_post_processed=[text])
    return chat_math_metric(4).sample_level_fn.compute(doc, resp)


def test_fusion_gold_wraps_only_bare_lists_and_tuples():
    assert fusion_aya_gold("1,3,9,11") == "$1,3,9,11$"
    assert fusion_aya_gold("(6,5)") == "$(6,5)$"
    assert fusion_aya_gold("864") == "864"
    assert fusion_aya_gold("4.25") == "4.25"
    assert fusion_aya_gold("$\\frac{4}{7}$") == "$\\frac{4}{7}$"


def test_fusion_chat_prompt_is_the_math_instruction_with_the_problem_in_its_language():
    line = {"question": "  Berechne $x$.  ", "answer": "5"}
    doc = _cfg("fusion_aya_math_chat:deu").prompt_function(line, "t")
    assert doc.query.endswith("Berechne $x$.") and "\\boxed{ANSWER}" in doc.query
    assert "Problem:" not in doc.query and "Solution:" not in doc.query and "Aufgabe" not in doc.query
    assert doc.choices == ["5"] and doc.gold_index == 0


def test_fusion_chat_scores_boxed_and_the_list_gold_as_a_set():
    line = {"question": "q", "answer": "1,3,9,11"}
    assert _score("fusion_aya_math_chat:eng", line, "so \\boxed{1, 11, 3, 9}")["avg@n"] == 1.0
    assert _score("fusion_aya_math_chat:eng", line, "the numbers are 1, 3, 9, 11")["avg@n"] == 0.0, "strict: no box, no credit"
    line = {"question": "q", "answer": "2400, 1350, 864"}
    assert _score("fusion_aya_math_chat:eng", line, "\\boxed{864}")["avg@n"] == 0.0, "the bare list is not its last number (version 1 rule)"


def test_fusion_chat_languages_revision_and_samples():
    names = [c.name for c in cme.TASKS_TABLE if c.name.startswith("fusion")]
    assert names == [f"fusion_aya_math_chat:{x}" for x in ("eng", "deu", "fra", "ukr")]
    assert all(c.hf_revision == "56c020a524e50e03df8b0cf7d4efbf8ae85a9d9c" for c in cme.TASKS_TABLE if c.name.startswith("fusion"))
    assert all(c.metrics[0].sample_level_fn.num_samples() == 4 for c in cme.TASKS_TABLE if c.name.startswith("fusion"))


def test_gsm8k_chat_gold_is_the_number_after_the_last_hash_marks():
    line = {"question": "How many?", "answer": "She has 1,200 + 34.\n#### 1,234"}
    doc = _cfg("gsm8k_chat").prompt_function(line, "t")
    assert doc.choices == ["1234"]
    assert _score("gsm8k_chat", line, "Total is 1234.\n\nTherefore, the final answer is: $\\boxed{1234}$.")["avg@n"] == 1.0
    assert _score("gsm8k_chat", line, "The total is 1234 people.")["avg@n"] == 0.0, "strict: a \\boxed answer is required"


def test_samples_per_prompt():
    assert _cfg("gsm8k_chat").metrics[0].sample_level_fn.num_samples() == 1
    assert _cfg("olympiadbench_chat").metrics[0].sample_level_fn.num_samples() == 4
    assert _cfg("hmmt_feb_2025_chat").metrics[0].sample_level_fn.num_samples() == 32


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
