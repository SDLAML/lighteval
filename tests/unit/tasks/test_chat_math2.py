"""Tests of the second set of chat evals (tasks/tasks/chat_math2.py): IMO-AnswerBench, OMEGA, MMLU-ProX (chat prompt).

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_math2.py`. No dataset is read here: the task table is checked for shape and
consistency (the FULL official sets are the default, owner rule 11-14; the opt-in subsets are tested in test_sampling.py); the scorers are the chat metrics of chat_evals / chat_metrics.
"""

import unittest

from lighteval.metrics.chat_metrics import chat_choice_metric, chat_math_metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_math2 as cm2
from lighteval.tasks.tasks.chat_evals import MATH_PROMPT


def _doc(gold_text: str) -> Doc:
    return Doc(query="q", choices=[cm2._boxed(gold_text)], gold_index=0, task_name="t")


def _resp(*texts):
    return ModelResponse(text=list(texts), text_post_processed=list(texts))


def test_boxed_gold():
    assert cm2._boxed("$\\frac{16}{27}$") == "\\boxed{\\frac{16}{27}}"
    assert cm2._boxed(" 12 ") == "\\boxed{12}"
    assert cm2._boxed("4, 27, 3125") == "\\boxed{4, 27, 3125}"
    assert cm2._boxed("$(2, \\frac{1}{2}, \\frac{9}{4})$") == "\\boxed{(2, \\frac{1}{2}, \\frac{9}{4})}"


def test_imo_prompt_is_the_math_500_instruction():
    line = {"Problem ID": "imo-bench-algebra-001", "Problem": "  Find $N$.\n", "Short Answer": "$3$", "Category": "Algebra", "Subcategory": "Operation"}
    d = cm2.imo_chat_prompt(line, "imo_answerbench_chat")
    assert d.query == MATH_PROMPT.format(problem="Find $N$.")
    assert d.choices == ["\\boxed{3}"] and d.gold_index == 0 and d.specific["problem_id"] == "imo-bench-algebra-001"


def test_imo_scoring_rules():
    m = chat_math_metric(1).sample_level_fn
    assert m.compute(_doc("12"), _resp("<think>r</think>\n\n\\boxed{12}"))["avg@n"] == 1.0
    assert m.compute(_doc("12"), _resp(""))["avg@n"] == 0.0
    assert m.compute(_doc("12"), _resp("\\boxed{12} or \\boxed{13}"))["avg@n"] == 0.0, "the LAST box is the answer"
    assert m.compute(_doc("\\frac{16}{27}"), _resp("\\boxed{\\dfrac{16}{27}}"))["avg@n"] == 1.0
    assert m.compute(_doc("f(x)=x+1"), _resp("\\boxed{x+1}"))["avg@n"] == 1.0, "a function gold accepts its right-hand side"


def test_omega_prompt_is_the_olmes_template():
    line = {"id": "algebra_func_area_test_in_000", "setting_key": "algebra_func_area", "ground_truth": "0.1",
            "messages": [{"role": "user", "content": "Find the area. Round to 1 decimal.\n\nPresent the answer in LaTex format: \\boxed{Your answer}"}]}
    d = cm2.omega_chat_prompt(line, "omega_chat:explorative:algebra_func_area:test_in")
    assert d.query.startswith("Find the area. Round to 1 decimal.\n\nShow your work and conclude with")
    assert "Present the answer" not in d.query and "\\boxed{6}" in d.query
    assert d.choices == ["\\boxed{0.1}"]


def test_omega_scoring_accepts_float_and_fraction():
    m = chat_math_metric(1).sample_level_fn
    line = {"id": "x", "ground_truth": "1.75", "messages": [{"role": "user", "content": "q"}]}
    gold = cm2.omega_chat_prompt(line, "t")
    assert m.compute(gold, _resp("Therefore, the final answer is \\boxed{7/4}.")) ["avg@n"] == 1.0
    assert m.compute(gold, _resp("the final answer is 1.75 but no box")) ["avg@n"] == 0.0, "strict needs the box"


def test_mmlu_prox_prompt_letters_and_gold():
    base = {"question_id": 70, "question": "Q?", "category": "law", "src": "x"}
    four = {**base, "answer_index": 3, **{f"option_{i}": v for i, v in enumerate(["a", "b", "c", "d"])}}
    for i in range(4, 10):
        four[f"option_{i}"] = None
    d = cm2.mmlu_prox_chat_prompt(four, "mmlu_prox_chat:de")
    assert d.choices == list("ABCD") and d.gold_index == 3 and "one of ABCD." in d.query and d.query.endswith("D) d")
    assert "\n\nQ?\n\nA) a\n" in d.query or "\n\nQ?\n\nA) a" in d.query
    ten = {**base, "answer_index": 8, **{f"option_{i}": chr(97 + i) for i in range(10)}}
    d10 = cm2.mmlu_prox_chat_prompt(ten, "t")
    assert d10.choices == list("ABCDEFGHIJ") and d10.gold_index == 8 and "one of ABCDEFGHIJ." in d10.query


def test_mmlu_prox_scoring_is_the_anchored_letter():
    m = chat_choice_metric(1).sample_level_fn
    d = Doc(query="q", choices=list("ABCD"), gold_index=2, task_name="t")
    assert m.compute(d, _resp("<think>r</think>\n\nAnswer: C"))["acc"] == 1.0
    assert m.compute(d, _resp("<think>r</think>\n\nAnswer: B or C"))["acc"] == 0.0


def test_omega_buckets_are_the_full_test_splits():
    assert {c: len(v) for c, v in cm2.OMEGA_SUBSETS.items()} == {"explorative": 41, "compositional": 7, "transformative": 7}
    assert all(len(set(v)) == len(v) for v in cm2.OMEGA_SUBSETS.values())
    assert {c: tuple(spec[2]) for c, spec in cm2.OMEGA_REPOS.items()} == {"explorative": ("test_in", "test_out"), "compositional": ("test",), "transformative": ("test",)}
    assert cm2.OMEGA_ITEMS == 9715 and cm2.MMLU_PROX_PER_LANG_ITEMS == 11759


def test_task_table_names_and_no_filters():
    names = [t.name for t in cm2.TASKS_TABLE]
    assert "imo_answerbench_chat" in names
    assert sum(n.startswith("omega_chat:") for n in names) == 96 and len(set(names)) == len(names)
    assert sum(n.startswith("mmlu_prox_chat:") for n in names) == 28 and "mmlu_prox_chat:en" not in names, "English is mmlu_pro_chat"
    assert all(t.generation_size == cm2.BUDGET for t in cm2.TASKS_TABLE)
    assert all(t.hf_filter is None for t in cm2.TASKS_TABLE), "the full sets: no subset filter without data.sample_per_group"
    assert {n.split(":")[1] for n in names if n.startswith("omega_chat:")} == {"explorative", "compositional", "transformative"}
    assert sum(n.startswith("omega_chat:explorative:") and n.endswith(":test_out") for n in names) == 41


def test_imo_csv_guard_refuses_a_swapped_file():
    import tempfile
    from pathlib import Path
    from unittest import mock

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "xdg" / "imo_bench"
        folder.mkdir(parents=True)
        (folder / cm2.IMO_FILE).write_text("Problem ID,Problem,Short Answer,Category,Subcategory,Source\n")
        with mock.patch.dict("os.environ", {"TITAN_EVAL_CACHE": tmp}):
            try:
                cm2._build_tasks()
            except ValueError as e:
                assert "is not the pinned" in str(e)
            else:
                raise AssertionError("a file that is not the pinned v2 must not load")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok  {name}")
            except unittest.SkipTest as e:
                print(f"skip {name}: {e}")
