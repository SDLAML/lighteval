"""Which tasks score on the answer cut at the first invented problem (titan_eval owner 11-17, metrics/math_cut.py), and which stop strings they have.

The base math generation tasks only: gsm8k, gsm_plus, gsm_symbolic:main, math_500, math:<subject>:gen, mgsm:<language>:gen, fusion_aya_math:<language>:gen (extractive_match, and for math_500 / math:gen also
extractive_match_boxed). Every other task keeps Metrics.expr_gold_metric / expr_gold_boxed_metric. Plain functions: run with pytest, or `python tests/unit/tasks/test_math_cut_tasks.py`.
"""

import re

from lighteval.metrics.math_cut import CutInventedProblem, cut_invented_problem
from lighteval.metrics.metrics import Metrics
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.multilingual.tasks import fusion_aya_math, mgsm
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import aime_hmmt_base, gsm8k, gsm_plus, gsm_symbolic, math, math_500
from lighteval.utils.language import Language


CUT = Metrics.expr_gold_cut_metric.value
CUT_BOXED = Metrics.expr_gold_boxed_cut_metric.value
PLAIN = Metrics.expr_gold_metric.value
PLAIN_BOXED = Metrics.expr_gold_boxed_metric.value

# every task of the fork that is scored on the cut text: the names the owner approved (11-17)
CUT_TASKS = re.compile(r"^(gsm8k|gsm_plus|gsm_symbolic:main|math_500|math:[a-z_]+:gen|mgsm:[a-z]{3}:gen|fusion_aya_math:[a-z]{3}:gen)$")


def _by_name(table):
    return {c.name: c for c in table}


def test_the_base_math_tasks_use_the_cut_metrics():
    assert gsm8k.gsm8k.metrics == (CUT,) and gsm_plus.gsm_plus.metrics == (CUT,)
    assert gsm8k.gsm8k.metrics[0] is CUT and gsm_plus.gsm_plus.metrics[0] is CUT
    assert math_500.math_500.metrics[0] is CUT and math_500.math_500.metrics[1] is CUT_BOXED and len(math_500.math_500.metrics) == 2
    symbolic = _by_name(gsm_symbolic.TASKS_TABLE)
    assert symbolic["gsm_symbolic:main"].metrics[0] is CUT and len(symbolic["gsm_symbolic:main"].metrics) == 1
    gen = [c for c in math.TASKS_TABLE if c.name.endswith(":gen")]
    assert len(gen) == 7 and all(c.metrics[0] is CUT and c.metrics[1] is CUT_BOXED and len(c.metrics) == 2 for c in gen)
    assert len(mgsm.TASKS_TABLE) == 40 and all(c.metrics[0] is CUT and len(c.metrics) == 1 for c in mgsm.TASKS_TABLE)
    assert len(fusion_aya_math.TASKS_TABLE) == 4 and all(c.metrics[0] is CUT and len(c.metrics) == 1 for c in fusion_aya_math.TASKS_TABLE)
    # the headline names are the ones the dashboards read
    for c in [gsm8k.gsm8k, gsm_plus.gsm_plus, math_500.math_500, *gen, *mgsm.TASKS_TABLE, *fusion_aya_math.TASKS_TABLE, symbolic["gsm_symbolic:main"]]:
        assert c.metrics[0].metric_name == "extractive_match" and isinstance(c.metrics[0].sample_level_fn, CutInventedProblem), c.name
    assert math_500.math_500.metrics[1].metric_name == "extractive_match_boxed" and all(c.metrics[1].metric_name == "extractive_match_boxed" for c in gen)


def test_versions_are_bumped():
    assert gsm8k.gsm8k.version == 3 and gsm_plus.gsm_plus.version == 2 and math_500.math_500.version == 4   # 11-19: +1 each (native digits + lone-variable rule)
    assert _by_name(gsm_symbolic.TASKS_TABLE)["gsm_symbolic:main"].version == 3 and _by_name(gsm_symbolic.TASKS_TABLE)["gsm_symbolic:p1"].version == 1
    assert all(c.version == 2 for c in math.TASKS_TABLE if c.name.endswith(":gen")) and all(c.version == 1 for c in math.TASKS_TABLE if c.name.endswith(":bpb"))
    assert all(c.version == 2 for c in mgsm.TASKS_TABLE) and all(c.version == 3 for c in fusion_aya_math.TASKS_TABLE)


def test_stop_strings():
    # "\nProblem:" is the label of these tasks' own prompts; the existing stops stay
    assert list(gsm8k.gsm8k.stop_sequence) == ["Question:", "\n####", "####", "\nProblem:"]
    assert list(gsm_plus.gsm_plus.stop_sequence) == ["Question:", "\n####", "####", "\nProblem:"]
    assert list(math_500.math_500.stop_sequence) == ["\nProblem:"]
    # already stopped at "Problem:" (they cut at the same place; nothing to add)
    assert list(_by_name(gsm_symbolic.TASKS_TABLE)["gsm_symbolic:main"].stop_sequence) == ["Problem:", "\n####", "####"]
    assert all(list(c.stop_sequence) == ["Problem:"] for c in math.TASKS_TABLE if c.name.endswith(":gen"))
    # MGSM and Fusion Aya Math: the stop strings are unchanged (the cut does the rest): the language's own prompt label, no "Вопрос:" for Russian
    assert mgsm.mgsm_stop_sequences(Language.RUSSIAN) == ["Question:", "Problem:", mgsm._MGSM_PROMPT_PARTS[Language.RUSSIAN][1]]
    assert "Вопрос:" not in mgsm.mgsm_stop_sequences(Language.RUSSIAN)
    assert fusion_aya_math.fusion_aya_stop_sequences(Language.ENGLISH) == ["\nProblem:", "\nProblem\n"]


def test_stopping_at_the_new_problem_stop_scores_like_the_cut_on_the_whole_text():
    doc = Doc(query="", choices=["7"], gold_index=0, task_name="x")
    for answer, metrics in (
        ("Step 1: 4 + 3 = 7\nSo the sum is 7.\n\nProblem:\nA 100 piece puzzle needs 12 pieces", (CUT,)),
        ("So \\boxed{7}.\n\nProblem:\nWhat is 6 + 6?\n\nSolution:\n\\boxed{12}", (CUT, CUT_BOXED)),
    ):
        stopped = answer[: answer.index("\nProblem:")]
        assert cut_invented_problem(answer) == stopped == cut_invented_problem(stopped)
        for metric in metrics:
            whole = metric.sample_level_fn.compute(doc, ModelResponse(text=[answer]))
            assert whole == 1.0 and whole == metric.sample_level_fn.compute(doc, ModelResponse(text=[stopped])), metric.metric_name
        assert PLAIN.sample_level_fn.compute(doc, ModelResponse(text=[answer])) == 0.0, "the plain scorer reads the invented problem"


def test_the_pipeline_applies_them_and_returns_the_headline_names():
    from lighteval.metrics import apply_metric

    answer = "4 + 3 = 7. So \\boxed{7}.\n\nProblem:\nWhat is 6 + 6?\nAnswer: \\boxed{12}"
    doc = gsm8k.gsm8k_prompt({"question": "What is 4 + 3?", "answer": "4 + 3 = 7 #### 7"}, "gsm8k")
    assert apply_metric([ModelResponse(text=[answer])], [doc], list(gsm8k.gsm8k.metrics)) == [{"extractive_match": 1.0}]
    assert apply_metric([ModelResponse(text=[answer])], [doc], [PLAIN]) == [{"extractive_match": 0.0}], "the plain metric reads the invented problem"
    out = apply_metric([ModelResponse(text=[answer])], [doc], list(math_500.math_500.metrics))
    assert out == [{"extractive_match": 1.0, "extractive_match_boxed": 1.0}], out
    out = apply_metric([ModelResponse(text=["So 7."])], [doc], list(math_500.math_500.metrics))
    assert out == [{"extractive_match": 1.0, "extractive_match_boxed": 0.0}], out
    # a language's own question word (MGSM, Russian): the generation goes on with "Вопрос:"
    ru = mgsm.mgsm_cot_prompt(Language.RUSSIAN)({"question": "Сколько будет 4 + 3?", "answer": "7"}, "mgsm:rus:gen")
    text = "4 + 3 = 7. Ответ: 7\n\nВопрос: сколько будет 6 + 6?\nОтвет: 12"
    config = next(c for c in mgsm.TASKS_TABLE if c.name == "mgsm:rus:gen")
    assert apply_metric([ModelResponse(text=[text])], [ru], list(config.metrics)) == [{"extractive_match": 1.0}]
    assert apply_metric([ModelResponse(text=[text])], [ru], [PLAIN]) == [{"extractive_match": 0.0}]


def test_tasks_outside_the_list_keep_the_plain_metrics():
    assert PLAIN is not CUT and PLAIN_BOXED is not CUT_BOXED
    # gsm_symbolic p1 / p2 were not part of the analysis
    symbolic = _by_name(gsm_symbolic.TASKS_TABLE)
    assert symbolic["gsm_symbolic:p1"].metrics == (PLAIN,) and symbolic["gsm_symbolic:p2"].metrics == (PLAIN,)
    assert symbolic["gsm_symbolic:p1"].version == 1 and symbolic["gsm_symbolic:p2"].version == 1
    # AIME / HMMT base: the plain scorer inside the pass@k metric (not part of the list)
    scorer = aime_hmmt_base.MathPassAt().compute_score.__self__
    assert not isinstance(scorer, CutInventedProblem) and type(scorer) is type(PLAIN.sample_level_fn)
    assert all(not any(isinstance(m.sample_level_fn, CutInventedProblem) for m in c.metrics) for c in aime_hmmt_base.TASKS_TABLE)
    # math:<subject>:bpb is not a generation task
    assert all(not any(isinstance(m.sample_level_fn, CutInventedProblem) for m in c.metrics) for c in math.TASKS_TABLE if c.name.endswith(":bpb"))


def test_no_other_task_of_the_fork_uses_the_cut():
    configs = Registry.load_all_task_configs(load_multilingual=True)
    cut = sorted(name for name, c in configs.items() if any(isinstance(m.sample_level_fn, CutInventedProblem) for m in c.metrics))
    assert cut and all(CUT_TASKS.match(name) for name in cut), [n for n in cut if not CUT_TASKS.match(n)]
    assert len(cut) == 1 + 1 + 1 + 1 + 7 + 40 + 4
    # the tasks that scored with the plain metric before, outside the list, still do
    for name, c in configs.items():
        if CUT_TASKS.match(name):
            continue
        for m in c.metrics:
            assert not isinstance(m.sample_level_fn, CutInventedProblem), name


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-q"]))
