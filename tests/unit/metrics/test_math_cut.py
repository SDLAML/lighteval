"""The invented-problem cut of the BASE math generation tasks (src/lighteval/metrics/math_cut.py, titan_eval owner 11-17).

The rule was analysed on the stored answers of every base model (analysis RULES["broad"], labels.json = fixture `math_cut_labels.json`, 107 labels); the real examples in
`math_cut_examples.json` are short pieces of stored answers (before / after the first invented-problem line) with the scorer's value on the whole text (stored) and on the cut text (new).
Plain functions: run with pytest, or `python tests/unit/metrics/test_math_cut.py`.
"""

import json
import random
import re
from pathlib import Path

from lighteval.metrics.math_cut import (
    EXTRA_LABELS,
    CutInventedProblem,
    cut_invented_problem,
    cut_position,
    invented_problem_labels,
    invented_problem_re,
    invented_problem_regex,
)
from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc


FIXTURES = Path(__file__).parent / "fixtures"
LABELS = json.loads((FIXTURES / "math_cut_labels.json").read_text())
EXAMPLES = json.loads((FIXTURES / "math_cut_examples.json").read_text())


def _reference_rx(labels):
    """The analysis' own `rx()` (tmp/p3/cutfix/analyse.py), copied as it was."""
    L = "|".join(sorted((re.escape(x) for x in labels), key=len, reverse=True))
    return re.compile(r"\n[ \t#*>\-]*(?:\d+[.)]\s*)?(?:\*\*)?(?:" + L + r"|第\s*\d+\s*题)\s*\d*\s*(?:\*\*)?\s*[:：]", re.I)


def _score(metric, gold, text):
    doc = Doc(query="", choices=[gold], gold_index=0, task_name="x")
    return float(metric.value.sample_level_fn.compute(doc, ModelResponse(text=[text])))


# ------------------------------------------------------------------------------------------------------------------------------- the labels and the rule
def test_labels_are_built_from_the_sources_and_equal_the_analysed_107():
    labels = invented_problem_labels()
    assert len(LABELS) == 107 and len(set(LABELS)) == 107
    assert len(labels) == 107 and set(labels) == set(LABELS)
    assert set(EXTRA_LABELS) <= set(labels)
    # one broad set for every task: the question word of every language and the problem label of the MGSM / Fusion Aya prompts are in it
    for word in ("вопрос", "pregunta", "frage", "question", "câu hỏi", "سؤال", "प्रश्न", "প্রশ্ন", "คำถาม", "質問", "问题", "문제"):
        assert word in labels, word
    for label in ("Bài toán", "Задача", "Aufgabe", "Problème", "ችግር", "المسألة", "সমস্যা"):
        assert label in labels, label
    assert not any(label.endswith((":", "：")) or label != label.strip() for label in labels)


def test_the_regex_is_the_analysed_one():
    assert invented_problem_re().pattern == _reference_rx(LABELS).pattern
    assert invented_problem_re().flags == _reference_rx(LABELS).flags
    assert invented_problem_regex(LABELS).pattern == _reference_rx(LABELS).pattern
    for example in EXAMPLES:
        text = example["before"] + example["after"]
        assert invented_problem_re().search(text).span() == _reference_rx(LABELS).search(text).span()


def test_real_examples_cut_where_the_analysis_cut_and_score_as_stored_and_new():
    assert 10 <= len(EXAMPLES) <= 20
    for example in EXAMPLES:
        text = example["before"] + example["after"]
        context = (example["task"], example["rule_hit"])
        position = cut_position(text)
        assert position == len(example["before"]), context
        assert cut_invented_problem(text) == example["before"], context
        assert invented_problem_re().search(text).group(0) == example["rule_hit"], context
        # the plain scorer on the whole text = the stored value, on the cut text = the new value; the cut metric on the whole text = the new value
        assert _score(Metrics.expr_gold_metric, example["gold"], text) == example["stored"], context
        assert _score(Metrics.expr_gold_metric, example["gold"], example["before"]) == example["new"], context
        assert _score(Metrics.expr_gold_cut_metric, example["gold"], text) == example["new"], context
    changes = {(e["stored"], e["new"]) for e in EXAMPLES}
    assert {(0, 1), (0, 0), (1, 1)} <= changes, "the examples cover wrong -> right, wrong -> wrong and right -> right"


def test_every_label_cuts_in_its_markdown_numbered_and_colon_variants():
    answer = "So the answer is 5."
    for label in invented_problem_labels():
        for form in (
            "\n{l}:", "\n{l} :", "\n{l}：", "\n{l} 2:", "\n{l}2:", "\n**{l}:", "\n**{l}:**", "\n**{l} 2:**", "\n**{l}**:", "\n### {l}:", "\n### **{l} 3:", "\n## {l} 4：",
            "\n- {l}:", "\n- **{l}**:", "\n> {l}:", "\n1. {l}:", "\n2) **{l}**:", "\n  {l}:", "\n\t{l}:", "\n{L}:", "\n{U}:",
        ):
            tail = form.format(l=label, L=label.upper(), U=label.lower()) + " the next one\nx = 7"
            assert cut_invented_problem(answer + tail) == answer, repr(tail)
            assert cut_position(answer + tail) == len(answer), repr(tail)
    # the chinese ordinal form "第N题"
    for tail in ("\n第2题：", "\n### 第 3 题：", "\n**第 12 题:**", "\n1. 第4题:"):
        assert cut_invented_problem(answer + tail + " ...") == answer, tail


def test_markdown_and_numbered_forms_of_the_stored_answers():
    cases = {
        "ok 4\n\nProblem:\nA 100 piece puzzle ...": "ok 4\n",
        "ok 4\n\n---\n\n**প্রশ্ন 2:**\nএকটি ...": "ok 4\n\n---\n",
        "ok 4\n\n### 问题 2：\n3 + 4": "ok 4\n",
        "ok 4\n\nPregunta: ¿Cuántos ...": "ok 4\n",
        "ok 4\n\nВопрос: Сколько": "ok 4\n",
        "ok 4\n3. **Question 3:** What": "ok 4",
        "ok 4\n\n#### Example 2:\nx": "ok 4\n",
        "ok 4\n- Question:\n  x": "ok 4",
        "ok 4\nQ: what is 3+4?": "ok 4",
        "ok 4\n q: what": "ok 4",
    }
    for text, expected in cases.items():
        assert cut_invented_problem(text) == expected, repr(text)


def test_no_cut_when_the_label_is_not_at_the_start_of_a_line_or_has_no_colon():
    texts = [
        "Let's start with Understanding the Problem: x = 3, so x + 4 = 7.",
        "1. **Understanding the Problem**\n   We need x.\n   The Question: what is x? It is 7.",
        "The answer to this Problem: 7",
        "Step 2. Problem: x = 7",                       # a label after other words on the line
        "see the Question: 7 and the Exercise: 8",
        "text\nProblem solved: 7",                      # the label is followed by another word, not by the colon
        "text\nProblems: 7",
        "text\nSubproblem: 7",
        "text\nQuestionable: 7",
        "text\nQuantity: 7",
        "text\nProblem\n7",                             # no colon
        "text\nProblem 7\nanswer",                      # a number but no colon
        "text\nExamples: 7",
        "Problem: 7 (the generation starts with the label, there is no line break before it)",
        "text \\\\ Problem: 7",
        "text\n7 Problem: 8",                           # a number before the label (the rule allows "N." or "N)" only)
        "text\n#1 Problem: 8",
        "",
    ]
    for text in texts:
        assert cut_position(text) is None, repr(text)
        assert cut_invented_problem(text) == text, repr(text)


def test_a_generation_that_starts_with_the_label_is_cut_to_nothing_and_scores_zero():
    assert cut_invented_problem("\nProblem: 12 + 30 = 42\nThe answer is 42") == ""
    assert cut_invented_problem("\n\nPregunta: 5") == "\n"
    assert cut_position("\nProblem: x") == 0
    text = "\nProblem: what is 6 + 7?\nSolution: 13\nThe final answer is 13"
    assert _score(Metrics.expr_gold_metric, "13", text) == 1.0           # the plain scorer reads the restated problem's solution
    assert _score(Metrics.expr_gold_cut_metric, "13", text) == 0.0       # nothing is left to read
    assert _score(Metrics.expr_gold_cut_metric, "13", "") == 0.0
    # right -> wrong exists and is the rule's price: the answer that came after a restated problem is gone
    box = "\n\nProblem: x\nSo \\boxed{13}"
    assert _score(Metrics.expr_gold_boxed_cut_metric, "13", box) == 0.0 and _score(Metrics.expr_gold_boxed_metric, "13", box) == 1.0


def test_the_first_match_wins_and_cutting_twice_changes_nothing():
    text = "a 1\n\nProblem: b 2\nSolution: 3\n\nProblem: c 4"
    assert cut_invented_problem(text) == "a 1\n"
    assert cut_invented_problem(cut_invented_problem(text)) == "a 1\n"
    rng = random.Random(1)
    pieces = ["\n", "Problem", ":", " ", "**", "#", "1", "2.", "Q", "x", "Pregunta", "：", "问题", "第", "题", "-", ">", "7", "Question", "\n\n"]
    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 24)))
        once = cut_invented_problem(text)
        assert cut_invented_problem(once) == once and text.startswith(once)


# --------------------------------------------------------------------------------------------------------------------------------- the metric wrapper
def test_the_new_metrics_keep_the_names_and_the_old_ones_are_unchanged():
    assert Metrics.expr_gold_cut_metric.value.metric_name == "extractive_match"
    assert Metrics.expr_gold_boxed_cut_metric.value.metric_name == "extractive_match_boxed"
    assert Metrics.expr_gold_metric.value.metric_name == "extractive_match"
    assert Metrics.expr_gold_boxed_metric.value.metric_name == "extractive_match_boxed"
    assert not isinstance(Metrics.expr_gold_metric.value.sample_level_fn, CutInventedProblem)
    assert not isinstance(Metrics.expr_gold_boxed_metric.value.sample_level_fn, CutInventedProblem)
    assert isinstance(Metrics.expr_gold_cut_metric.value.sample_level_fn, CutInventedProblem)
    assert isinstance(Metrics.expr_gold_boxed_cut_metric.value.sample_level_fn, CutInventedProblem)
    plain, cut = Metrics.expr_gold_metric.value, Metrics.expr_gold_cut_metric.value
    inner = cut.sample_level_fn.metric
    assert inner is not plain.sample_level_fn                      # its own copy
    for attribute in ("language", "fallback_mode", "precision", "aggregation_function", "timeout_seconds"):
        assert getattr(inner, attribute) == getattr(plain.sample_level_fn, attribute), attribute
    assert type(inner) is type(plain.sample_level_fn) and cut.higher_is_better and cut.corpus_level_fn is plain.corpus_level_fn
    assert type(Metrics.expr_gold_boxed_cut_metric.value.sample_level_fn.metric) is type(Metrics.expr_gold_boxed_metric.value.sample_level_fn)
    # the plain metrics still read the invented problem (other tasks use them)
    text = "So 4 + 3 = 7.\n\nProblem:\nWhat is 6 + 6?\nAnswer: 12"
    assert _score(Metrics.expr_gold_metric, "7", text) == 0.0 and _score(Metrics.expr_gold_cut_metric, "7", text) == 1.0


def test_the_wrapper_is_the_wrapped_metric_on_the_cut_text():
    gold = Doc(query="", choices=["7"], gold_index=0, task_name="x")
    plain = Metrics.expr_gold_metric.value.sample_level_fn
    for text in ("So 4 + 3 = 7.\n\nProblem:\nWhat is 6 + 6?\nAnswer: 12", "7", "so 8", "", "\\boxed{7}\n\nQuestion: \\boxed{9}"):
        cut = cut_invented_problem(text)
        assert Metrics.expr_gold_cut_metric.value.sample_level_fn.compute(gold, ModelResponse(text=[text])) == plain.compute(gold, ModelResponse(text=[cut]))
    # the stored text is not changed, and the post-processed text is cut too
    response = ModelResponse(text=["4 + 3 = 7\n\nProblem: 6 + 6 = 12"])
    Metrics.expr_gold_cut_metric.value.sample_level_fn.compute(gold, response)
    assert response.text == ["4 + 3 = 7\n\nProblem: 6 + 6 = 12"] and response.text_post_processed is None
    processed = ModelResponse(text=["raw 1"], text_post_processed=["4 + 3 = 7\n\nProblem: 6 + 6 = 12"])
    assert Metrics.expr_gold_cut_metric.value.sample_level_fn.compute(gold, processed) == 1.0
    assert processed.text_post_processed == ["4 + 3 = 7\n\nProblem: 6 + 6 = 12"]
    # the boxed twin: a box only in the invented problem scores 0, a box before it scores as the plain metric
    assert _score(Metrics.expr_gold_boxed_cut_metric, "7", "So \\boxed{7}.\n\nProblem: \\boxed{9}") == 1.0
    assert _score(Metrics.expr_gold_boxed_cut_metric, "7", "So 7.\n\nProblem: \\boxed{7}") == 0.0
    assert _score(Metrics.expr_gold_boxed_metric, "7", "So 7.\n\nProblem: \\boxed{7}") == 1.0


# ------------------------------------------------------------------------------------------------------------------------ the stop string "\nProblem:"
def _stop_at(text, stops):
    found = [text.find(stop) for stop in stops if stop in text]
    return text[: min(found)] if found else text


def test_the_problem_stop_gives_the_same_scored_text_as_the_cut_without_it():
    stop = ["\nProblem:"]
    for example in EXAMPLES:
        text = example["before"] + example["after"]
        assert cut_invented_problem(_stop_at(text, stop)) == cut_invented_problem(text), example["rule_hit"]
        assert _score(Metrics.expr_gold_cut_metric, example["gold"], _stop_at(text, stop)) == _score(Metrics.expr_gold_cut_metric, example["gold"], text)
    assert cut_invented_problem(_stop_at("So 7.\n\nProblem:\nx 9", stop)) == cut_invented_problem("So 7.\n\nProblem:\nx 9") == "So 7.\n"
    # random texts made of the rule's own parts: the same text, with ONE exception (next test)
    rng = random.Random(0)
    pieces = ["\n", "Problem", "Problem:", "\nProblem:", ":", " ", "**", "#", "1", "2.", "Q", "x", "Pregunta", "：", "问题", "第", "题", "-", ">", "7", "Question", "\n\n", "\t"]
    exceptions = 0
    for _ in range(20000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 30)))
        stopped = _stop_at(text, stop)
        if cut_invented_problem(stopped) == cut_invented_problem(text):
            continue
        exceptions += 1
        assert _only_a_list_marker_line_in_between(text, stopped), repr(text)
    assert exceptions < 100       # a few of 20,000 random strings: the exception is rare even among strings built from its own parts


def _only_a_list_marker_line_in_between(text, stopped):
    """The one place where the rule cuts earlier than the stop: a list marker line ("\\n2.", "\\n3)") directly before "\\nProblem:". The rule's `\\d+[.)]\\s*` may take the line break that
    precedes "Problem:", so the match starts at the marker's line; the stop string starts at the line break after the marker, and the marker line stays in the stopped text."""
    cut = cut_invented_problem(text)
    return text.startswith(stopped) and stopped.startswith(cut) and re.fullmatch(r"\n[ \t#*>\-]*\d+[.)]\s*", stopped[len(cut):]) is not None


def test_the_one_difference_between_the_stop_and_the_cut_is_a_bare_list_marker_line():
    stop = ["\nProblem:"]
    text = "So 7.\n2.\nProblem:\nx 9"
    assert cut_invented_problem(text) == "So 7."                        # the rule's match starts at the marker line "\n2." and takes the line break after it
    assert cut_invented_problem(_stop_at(text, stop)) == "So 7.\n2."    # the stopped text ends with the marker (no label left to match)
    assert _only_a_list_marker_line_in_between(text, _stop_at(text, stop))
    for same in ("So 7.\n2. Problem:\nx 9", "So 7.\n2.5\nProblem:\nx 9", "So 7.\nstep 2.\nProblem:\nx 9"):
        assert cut_invented_problem(_stop_at(same, stop)) == cut_invented_problem(same), same


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-q"]))
