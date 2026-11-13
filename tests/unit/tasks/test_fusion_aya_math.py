"""Fusion Aya Math Bench (fusion_aya_math:<lang>:gen), version 1 (11-13 review fixes): stop strings at the start of a line, list / tuple golds
written as $...$. Plain functions without fixtures: run with pytest, or `python tests/unit/tasks/test_fusion_aya_math.py`.
"""

import re

from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.multilingual.tasks import fusion_aya_math as fam
from lighteval.utils.language import Language


def _cut_at_stop(text, stops):
    found = [text.find(stop) for stop in stops if stop in text]
    return text[: min(found)] if found else text


def _score(gold, prediction, language=Language.ENGLISH):
    doc = fam.fusion_aya_cot_prompt(language)({"question": "q", "answer": gold}, "t")
    return Metrics.expr_gold_metric.value.sample_level_fn.compute(doc, ModelResponse(text=[prediction]))


def test_stop_strings_are_line_anchored_and_the_colon_less_aliases_only_whole_lines():
    assert fam.fusion_aya_stop_sequences(Language.ENGLISH) == ["\nProblem:", "\nProblem\n"]
    assert fam.fusion_aya_stop_sequences(Language.GERMAN) == ["\nProblem:", "\nAufgabe:", "\nAufgabe\n"]
    assert fam.fusion_aya_stop_sequences(Language.FRENCH) == ["\nProblem:", "\nProblème:", "\nProblème\n"]
    assert fam.fusion_aya_stop_sequences(Language.UKRAINIAN) == ["\nProblem:", "\nЗадача:", "\nЗадача\n"]
    old = {
        Language.ENGLISH: ["Problem:", "Problem:", "Problem"],
        Language.GERMAN: ["Problem:", "Aufgabe:", "Aufgabe"],
    }
    own = {  # the model's own text; version 0 cut every one of them at the word
        Language.ENGLISH: ["Let's go.\n\n1. **Understanding the Problem**\n   We need ...", "### Step 1: Understanding the Problem:\nx = 3"],
        Language.GERMAN: ["Um die Aufgabe zu lösen, setzen wir x = 3.", "1. **Ziel der Aufgabe**\n   Wir suchen ..."],
        Language.FRENCH: ["Pour résoudre ce Problème, posons x = 3."],
        Language.UKRAINIAN: ["Щоб розв'язати цю Задача, нехай x = 3."],
    }
    restarts = {
        Language.ENGLISH: ["\n\nProblem:\nFind ...", "\nProblem\nFind ..."],
        Language.GERMAN: ["\n\nAufgabe:\nBestimme ...", "\nAufgabe\nBestimme ...", "\n\nProblem:\nFind ..."],
        Language.FRENCH: ["\n\nProblème:\nTrouve ...", "\nProblème\nTrouve ..."],
        Language.UKRAINIAN: ["\n\nЗадача:\nЗнайди ...", "\nЗадача\nЗнайди ..."],
    }
    for language, texts in own.items():
        stops = fam.fusion_aya_stop_sequences(language)
        for text in texts:
            assert _cut_at_stop(text, stops) == text, (language, text)
    for language, tails in restarts.items():
        stops = fam.fusion_aya_stop_sequences(language)
        for tail in tails:
            # the stop string starts at the last "\n" before the label, so a blank line leaves one "\n" behind
            assert _cut_at_stop("So the answer is 7." + tail, stops).rstrip("\n") == "So the answer is 7.", (language, tail)
    assert _cut_at_stop(own[Language.ENGLISH][0], old[Language.ENGLISH]) != own[Language.ENGLISH][0]  # the old stops cut it
    assert _cut_at_stop(own[Language.GERMAN][0], old[Language.GERMAN]) != own[Language.GERMAN][0]
    assert all(c.version == 1 for c in fam.TASKS_TABLE) and len(fam.TASKS_TABLE) == 4


def test_gold_is_written_as_latex_only_for_bare_lists_and_tuples():
    cases = {
        "1,3,5": "$1,3,5$",
        "2400, 1350, 864": "$2400, 1350, 864$",
        "-3,0": "$-3,0$",
        "(6,5)": "$(6,5)$",
        "(1,-4,-2),(3,2,3),(13,2,-2)": "$(1,-4,-2),(3,2,3),(13,2,-2)$",
        "(7,5,1,1,9,3,5,7,3,8,6,4,2,9,2,4,6,8)": "$(7,5,1,1,9,3,5,7,3,8,6,4,2,9,2,4,6,8)$",
        # unchanged
        "73": "73", "-5": "-5", "4.25": "4.25", "$\\frac{4}{7}$": "$\\frac{4}{7}$", "$64,\\frac{1}{4}$": "$64,\\frac{1}{4}$",
        "$a=2$, $a=-6-4 \\sqrt{2}$": "$a=2$, $a=-6-4 \\sqrt{2}$", "$(0,2),(0,-2),(4,23),(4,-23)$": "$(0,2),(0,-2),(4,23),(4,-23)$",
    }
    for gold, expected in cases.items():
        assert fam.fusion_aya_gold(gold) == expected, gold
        doc = fam.fusion_aya_cot_prompt(Language.GERMAN)({"question": "q", "answer": gold}, "t")
        assert doc.get_golds() == [expected] and doc.query.endswith("\n\nLösung:")  # the prompt is the one of version 0


def test_list_golds_are_sets_and_tuple_golds_are_ordered():
    box = lambda a: f"So the answer is $\\boxed{{{a}}}$."  # noqa: E731
    # a list: any order, nothing missing, nothing extra, no read as the decimal 1.3
    assert _score("1,3,9,11", box("1, 11, 3, 9")) == 1 and _score("1,3,9,11", box("1,3,9,11")) == 1
    assert _score("1,3,9,11", box("1,3,9")) == 0 and _score("1,3,9,11", box("1,3,9,11,13")) == 0 and _score("1,3,5", box("1.3")) == 0
    assert _score("2400, 1350, 864", box("864, 1350, 2400")) == 1
    assert _score("2400, 1350, 864", box("864")) == 0  # version 0 read the gold as 864 and scored this 1
    assert _score("-3,0", box("0, -3")) == 1 and _score("-3,0", box("-3")) == 0
    # a tuple: ordered; a list of tuples: a set of ordered tuples
    assert _score("(6,5)", box("(6,5)")) == 1 and _score("(6,5)", box("(6, 5)")) == 1
    assert _score("(6,5)", box("(5,6)")) == 0 and _score("(6,5)", box("(9,8)")) == 0
    assert _score("(1,-4,-2),(3,2,3),(13,2,-2)", box("(13,2,-2),(1,-4,-2),(3,2,3)")) == 1
    assert _score("(1,-4,-2),(3,2,3),(13,2,-2)", box("(1,-4,-2),(3,2,3)")) == 0
    assert _score("(1,-4,-2),(3,2,3),(13,2,-2)", box("(1,4,-2),(3,2,3),(13,2,-2)")) == 0
    long_tuple = "(7,5,1,1,9,3,5,7,3,8,6,4,2,9,2,4,6,8)"
    assert _score(long_tuple, box(long_tuple)) == 1 and _score(long_tuple, box("(5,7,1,1,9,3,5,7,3,8,6,4,2,9,2,4,6,8)")) == 0
    # unchanged golds: integers, decimals ("4.25" always worked), LaTeX
    assert _score("4.25", box("4.25")) == 1 and _score("4.25", "The value is 4.250") == 1 and _score("4.25", box("\\frac{17}{4}")) == 1
    assert _score("4.25", box("4.26")) == 0
    assert _score("73", box("73")) == 1 and _score("73", box("74")) == 0
    assert _score("$\\frac{4}{7}$", box("\\frac{4}{7}")) == 1 and _score("$64,\\frac{1}{4}$", box("\\frac{1}{4}, 64")) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
