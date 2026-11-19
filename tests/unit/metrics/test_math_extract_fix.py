"""Extraction fixes of the BASE math tasks (titan_eval 11-19, extraction audit): native digits and a bare LaTeX variable picked as the answer.

Both live in the wrapper `CutInventedProblem` (metrics/math_cut.py) that gsm8k, gsm_plus, gsm_symbolic, math_500, math:gen, mgsm and fusion_aya_math score through; `Metrics.expr_gold_metric` (every other
task) is unchanged. The failing examples are pieces of stored answers (Qwen3.5-35B-A3B-Base mgsm:ben id 113; gemma-4-26B-A4B math_500 ids 199, 376, 342, 87, 191; math:gen), each scored 0 before.
Plain functions: run with pytest, or `python tests/unit/metrics/test_math_extract_fix.py`.
"""

from lighteval.metrics.math_cut import ascii_digits
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


def _score(metric, gold, text):
    doc = Doc(query="", choices=[gold], gold_index=0, task_name="x")
    return float(metric.value.sample_level_fn.compute(doc, ModelResponse(text=[text])))


def _extract(text, skip):
    doc = Doc(query="", choices=["x"], gold_index=0, task_name="x")
    regexes = get_extraction_regexes(doc, (ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)), Language.ENGLISH)
    return [str(p) for p in extract_target_from_pred(text, regexes, "first_match", "any_match", 5, skip_lone_variable=skip)]


# ------------------------------------------------------------------------------------------------------------------------------------ native digits
def test_ascii_digits_maps_every_decimal_digit_script():
    for native in ("২৫", "२५", "٢٥", "۲۵", "၂၅", "២៥", "２５", "๒๕"):            # Bengali, Devanagari, Arabic-Indic, Persian, Burmese, Khmer, fullwidth, Thai
        assert ascii_digits(native) == "25", native
    assert ascii_digits("৳১,২০০ টাকা") == "৳1,200 টাকা"
    assert ascii_digits("١٢٫٥ و ١٬٠٠٠") == "12.5 و 1,000"          # Arabic decimal and thousands separators
    assert ascii_digits("x² + ½ = ①") == "x² + ½ = ①"                  # superscripts, fractions, circled numbers are not decimal digits (category No)
    text = "plain 25 text"
    assert ascii_digits(text) is text


def test_native_digit_answers_score_in_the_base_math_metric_only():
    # Qwen3.5-35B-A3B-Base, mgsm:ben id 113, gold 25: the answer "উত্তর: ২৫" was scored 0
    for text, gold in (("রায়ানের বাগানে মোট ২৫ টি ফুল থাকবে।\n\nউত্তর: ২৫", "25"),
                       ("समाधान:\nरामले १०० रुपैयाँ बच्छ।\nउनीले २५ रुपैयाँ", "25"),
                       ("الجواب: ٢٥", "25"),
                       ("ဖြေ: ၂၅", "25"),
                       ("ចម្លើយ៖ ២៥", "25")):
        assert _score(Metrics.expr_gold_cut_metric, gold, text) == 1.0, text
        assert _score(Metrics.expr_gold_metric, gold, text) == 0.0, text       # the plain scorer of the other tasks is unchanged
    assert _score(Metrics.expr_gold_cut_metric, "25", "উত্তর: ২৬") == 0.0       # a wrong native number stays wrong
    assert _score(Metrics.expr_gold_cut_metric, "1200", "উত্তর: ১,২০০") == 1.0
    assert _score(Metrics.expr_gold_cut_metric, "12.5", "الجواب: ١٢٫٥") == 1.0


def test_the_digits_are_mapped_after_the_invented_problem_cut():
    text = "উত্তর: ২৫\n\nপ্রশ্ন ২: আরও ১০০ টাকা কত?\n\nসমাধান: ১০০"        # the invented problem's number (100) must not be read
    assert _score(Metrics.expr_gold_cut_metric, "25", text) == 1.0


# ------------------------------------------------------------------------------------------------------------------------- a bare variable is not the answer
GOLD_BOXED = {"17": "so $x=\\boxed{17}$.", "2": "so $\\boxed{2}$.", "(8,-2)": "so $P=\\boxed{(8,-2)}$.", "0": "so $\\boxed{0}$.", "\\frac43": "so $\\boxed{\\frac43}$."}


def test_lone_variable_followed_by_the_answer_on_its_line_is_skipped():
    # gemma-4-26B-A4B, math_500 id 199 / 376 / 342 / 191 (all scored 0 with the extracted prediction x / P / x / b)
    cases = (
        ("$x = 17$\n\nTherefore, the value of $x$ is 17.\n\nAnswer:\nThe value of $x$ is 17.", "17"),
        ("Therefore, the coordinates of point $P$ are $(8, -2)$.\n\nFinal Answer:\nThe coordinates of point $P$ are $(8, -2)$.", "(8,-2)"),
        ("is 2.\n\nFinal Answer:\nThe integer $x$ that satisfies $\\frac{1}{4}<\\frac{x}{7}<\\frac{1}{3}$ is 2.", "2"),
        ("Therefore, when $a = 4$, the value of $b$ is 0.\n\nAnswer:\nThe value of $b$ when $a = 4$ is 0.", "0"),
        ("Answer:\nThe horizontal asymptote is $y$ = $\\frac{4}{3}$.", "\\frac43"),
    )
    for text, gold in cases:
        assert _score(Metrics.expr_gold_metric, GOLD_BOXED[gold], text) == 0.0, text          # the plain scorer reads the variable
        assert _score(Metrics.expr_gold_cut_metric, GOLD_BOXED[gold], text) == 1.0, text


def test_extraction_with_and_without_the_rule():
    text = "Final Answer:\nThe value of $x$ is 17."
    assert _extract(text, skip=False)[0] == "x"
    assert _extract(text, skip=True)[0] == "17"
    text = "Final Answer:\nThe coordinates of point $P$ are $(8, -2)$."
    assert _extract(text, skip=False)[0] == "P"
    assert _extract(text, skip=True)[0] == "(8, -2)"
    for variable in ("$x_1$", "$x_{12}$", "$P'$", "\\(n\\)", "$ k $"):
        assert _extract(f"Answer:\nThe value of {variable} is 5 m.", skip=False)[0] != "5", variable
        assert _extract(f"Answer:\nThe value of {variable} is 5 m.", skip=True)[0] == "5", variable
    # on one line the number already won (the number pattern reaches it): both read 5
    assert _extract("Answer: the value of $x$ is 5 m.", skip=False)[0] == _extract("Answer: the value of $x$ is 5 m.", skip=True)[0] == "5"


def test_the_rule_leaves_every_other_extraction_alone():
    # nothing follows the variable on its line: it stays the answer (a letter answer such as "n" or the last-line variable)
    assert _extract("Final Answer: $x$", skip=True)[0] == "x"
    assert _extract("The answer is $k$.\nThanks.", skip=True)[0] == "k"
    # a variable that ends its line is not skipped because of the NEXT line (the pattern's trailing `\s*` swallows the newline: the rule looks at the rest of the line after the closing delimiter)
    assert _extract("Final Answer:\n$x$\n\nThe value is 17.", skip=True) == _extract("Final Answer:\n$x$\n\nThe value is 17.", skip=False)
    assert _extract("Final Answer:\n$x$\n\nThe value is 17.", skip=True)[0] == "x"
    # the number BEFORE the variable is not touched, and neither is a variable inside a longer expression
    assert _extract("Final Answer:\nThere are 8 integer values of $t$.", skip=True) == _extract("Final Answer:\nThere are 8 integer values of $t$.", skip=False)
    assert _extract("Final Answer:\nThe value of $x + y$ is 17.", skip=True) == _extract("Final Answer:\nThe value of $x + y$ is 17.", skip=False)
    assert _extract("Answer: $2x$ and 5", skip=True) == _extract("Answer: $2x$ and 5", skip=False)
    # a box is never skipped, whatever follows
    assert _extract("So \\boxed{x} since 3 > 2.", skip=True) == _extract("So \\boxed{x} since 3 > 2.", skip=False)
    assert _score(Metrics.expr_gold_cut_metric, "so $\\boxed{x}$.", "So $\\boxed{x}$ because 5 > 3.") == 1.0
    # a letter gold with a prose answer that names a number later is the case the rule gives up: documented, not silent
    assert _score(Metrics.expr_gold_cut_metric, "so $\\boxed{x}$.", "The answer is $x$.") == 1.0


def test_only_the_base_math_scorers_have_the_rule():
    assert Metrics.expr_gold_metric.value.sample_level_fn.skip_lone_variable is False
    assert Metrics.expr_gold_boxed_metric.value.sample_level_fn.skip_lone_variable is False
    assert Metrics.expr_gold_cut_metric.value.sample_level_fn.metric.skip_lone_variable is True
    assert Metrics.expr_gold_boxed_cut_metric.value.sample_level_fn.metric.skip_lone_variable is True
    assert Metrics.expr_gold_cut_metric.value.sample_level_fn.metric is not Metrics.expr_gold_metric.value.sample_level_fn


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
