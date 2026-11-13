"""Tests of the titan_eval chat evals: the strict reasoning split, the pipeline's --strict-reasoning, the chat metrics, the chat prompts, the
VLLMModelConfig fields for chat runs and the `num_samples` override.

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_evals.py`. Nothing here needs a GPU or a dataset except the tests that
read the pinned datasets (they skip when those are not in the Hugging Face cache).
"""

import unittest

from lighteval.metrics.chat_metrics import chat_choice_metric, chat_math_metric, unbiased_pass_at_k
from lighteval.models.model_output import ModelResponse
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_evals
from lighteval.utils.utils import remove_reasoning_tags, split_reasoning


TAGS = [("<think>", "</think>")]

# (text, context, answer, status): the table the titan_eval side (titan_eval/evals/reasoning.py) runs too (tests/test_chat_eval.py)
SPLIT_CASES = [
    ("<think>reason</think>\n\nThe answer is 5.", None, "The answer is 5.", "closed"),
    ("<think>a</think> x <think>b</think> final", None, "final", "closed"),
    ("<think>reason \\boxed{7} still going", None, "", "unclosed"),                       # unclosed: no answer, the mid-trace box must not score
    ("reason \\boxed{7} still going", "<|im_start|>assistant\n<think>\n", "", "unclosed"),   # prompt ends inside the block, no end tag
    ("reason</think>\n\n\\boxed{7}", "<|im_start|>assistant\n<think>\n", "\\boxed{7}", "closed"),
    ("plain answer", "<|im_start|>assistant\n<think>\n\n</think>\n\n", "plain answer", "absent"),   # thinking off: the empty block is in the prompt
    ("plain answer", None, "plain answer", "absent"),
    ("", None, "", "absent"),
    ("<think>r</think>", None, "", "closed"),                                             # closed, nothing after it
    ("<think>r</think> a <think> new block never closed", None, "", "unclosed"),
    ("no start </think> after", None, "after", "closed"),                                 # an end tag alone closes (the start was in the prompt or cut)
]


def test_split_reasoning_table():
    for text, ctx, answer, status in SPLIT_CASES:
        assert split_reasoning(text, TAGS, ctx) == (answer, status), (text, ctx, split_reasoning(text, TAGS, ctx))


def test_remove_reasoning_tags_keeps_its_upstream_behaviour():
    # the base path: an unclosed block stays in the text (this is why chat runs use split_reasoning through --strict-reasoning)
    assert remove_reasoning_tags("<think> r </think> A", TAGS) == " A"
    assert remove_reasoning_tags("<think> unclosed \\boxed{7}", TAGS) == "<think> unclosed \\boxed{7}"


def _pipeline(strict: bool) -> Pipeline:
    p = object.__new__(Pipeline)
    p.pipeline_parameters = PipelineParameters(launcher_type=ParallelismManager.VLLM, strict_reasoning=strict)
    return p


def _responses():
    prompt_in = "<|im_start|>assistant\n<think>\n"
    return [ModelResponse(input=prompt_in, text=["r1 </think>\n\nAnswer: B", "r2 never closes \\boxed{7}"]),
            ModelResponse(input="<|im_start|>assistant\n", text=["<think>x</think>\n\nok", "<think>y no end"])]


def test_pipeline_strict_reasoning_vs_upstream():
    strict, plain = _responses(), _responses()
    _pipeline(True)._post_process_outputs({"GENERATIVE": strict})
    _pipeline(False)._post_process_outputs({"GENERATIVE": plain})
    assert [r.text_post_processed for r in strict] == [["Answer: B", ""], ["ok", ""]]
    # upstream: the context-aware removal handles the first, the unclosed ones stay in the scored text
    assert plain[0].text_post_processed[0] == "\n\nAnswer: B" and "\\boxed{7}" in plain[0].text_post_processed[1]
    assert plain[1].text_post_processed == ["\n\nok", "<think>y no end"]
    assert PipelineParameters(launcher_type=ParallelismManager.VLLM).strict_reasoning is False, "the default stays the upstream behaviour"


def test_cli_has_the_strict_reasoning_flag_and_the_old_ones():
    import inspect

    from typer.testing import CliRunner

    from lighteval import cli_args, main_vllm
    from lighteval.__main__ import app
    assert cli_args.strict_reasoning.default is False and cli_args.remove_reasoning_tags.default is True and cli_args.reasoning_tags.default
    params = inspect.signature(main_vllm.vllm).parameters
    assert {"strict_reasoning", "remove_reasoning_tags", "reasoning_tags"} <= set(params)
    out = CliRunner().invoke(app, ["vllm", "--help"], terminal_width=240).output
    assert "--strict-reasoning" in out, out[-800:]


# ---- metrics -----------------------------------------------------------------------------------------------------------------

def _math_doc(gold="204"):
    return Doc(query="q", choices=[gold], gold_index=0, task_name="t")


def _resp(*texts):
    return ModelResponse(text=list(texts), text_post_processed=list(texts))


def test_chat_math_metric_strict_lenient_and_pass_at_k():
    m = chat_math_metric(4).sample_level_fn
    out = m.compute(_math_doc(), _resp("so \\boxed{204}", "the answer is 204.", "\\boxed{7}", ""))
    assert out["avg@n"] == 0.25, "only the boxed correct one counts in the strict number"
    assert out["avg@n_lenient"] == 0.5, "an unboxed last number counts in the lenient number only"
    assert abs(out["pass@2"] - (1 - 3 / 6)) < 1e-9 and out["pass@4"] == 1.0 and "pass@8" not in out, "pass@k only for k <= n"
    assert m.compute(_math_doc("070"), _resp("\\boxed{70}"))["avg@n"] == 1.0, "zero-padded AIME gold = the number"
    assert m.compute(_math_doc(), _resp("a \\boxed{1} b \\boxed{204}"))["avg@n"] == 1.0, "the LAST box"
    assert m.compute(_math_doc(), _resp("a \\boxed{204} b \\boxed{1}"))["avg@n"] == 0.0
    assert m.num_samples() == 4


def test_chat_math_metric_reads_the_post_processed_text_only():
    m = chat_math_metric(1).sample_level_fn
    r = ModelResponse(text=["<think>\\boxed{204}"], text_post_processed=[""])      # unclosed block: the strict pipeline gave an empty answer
    assert m.compute(_math_doc(), r)["avg@n"] == 0.0 and m.compute(_math_doc(), r)["avg@n_lenient"] == 0.0


def _mc_doc(gold=2, n=4):
    letters = list("ABCDEFGHIJ")[:n]
    return Doc(query="q", choices=letters, gold_index=gold, task_name="t")


def test_chat_choice_metric_anchored_vs_unanchored():
    m = chat_choice_metric(1).sample_level_fn
    ok = m.compute(_mc_doc(), _resp("Reasoning ... Answer: C"))
    assert ok["acc"] == 1.0 and ok["acc_unanchored"] == 1.0
    bad = m.compute(_mc_doc(), _resp("Answer: B"))
    assert bad["acc"] == 0.0 and bad["acc_unanchored"] == 0.0
    # a stray letter without any anchor: the upstream behaviour (unanchored) credits it, the anchored number must not
    stray = m.compute(_mc_doc(), _resp("I considered all of them and it is clearly\nC"))
    assert stray["acc"] == 0.0 and stray["acc_unanchored"] == 1.0
    empty = m.compute(_mc_doc(), _resp(""))
    assert empty["acc"] == 0.0 and empty["acc_unanchored"] == 0.0
    ten = m.compute(_mc_doc(gold=8, n=10), _resp("so Answer: I"))
    assert ten["acc"] == 1.0, "letters up to J are valid for 10 options"
    assert m.compute(_mc_doc(gold=2, n=4), _resp("Answer: E"))["acc"] == 0.0, "a letter outside the options is no answer"
    many = chat_choice_metric(8).sample_level_fn.compute(_mc_doc(), _resp(*(["Answer: C"] * 3 + ["Answer: A"] * 5)))
    assert many["acc"] == 3 / 8 and abs(many["pass@2"] - (1 - (10 / 28))) < 1e-9


def test_anchored_letter_variants():
    from lighteval.metrics.chat_metrics import anchored_letter
    hits = ["Answer: C", "**Answer: C**", "**Answer:** C", "Answer: $C$", "Answer: $C", "Answer: \\boxed{C}", "Answer: $\\text{C}$", "Answer:C", "Answer: C) text",
            "Answer: (C) text", "The answer is (C).", "Therefore, the answer is C", "ANSWER: C", "Final answer: C", "answer is C"]
    for h in hits:
        assert anchored_letter("reasoning\n" + h, 4) == "C", h
    misses = ["C", "C.", "Option C is correct", "The correct option is C.", "answer: c", "Answer:\nC", "A. foo\nB. bar\nC. baz", "Answer: $LETTER", "answer the question"]
    for x in misses:
        assert anchored_letter(x, 4) is None, x
    assert anchored_letter("Answer: A\nC. later line", 4) == "A", "a later line that starts with a letter is not an answer"
    assert anchored_letter("Answer: A ... on reflection Answer: C", 4) == "C", "the LAST anchored answer"
    assert anchored_letter("Answer: E", 4) is None and anchored_letter("Answer: E", 5) == "E"
    # a hedge is no answer: naming the gold letter first must not earn credit (multiple_choice.md)
    for h in ("Answer: B or C", "Answer: C, A", "Answer: B and C", "Answer: B/C", "Answer: \\boxed{B \\text{ or } C}", "Answer: C & D", "Answer: (B) or (C)"):
        assert anchored_letter(h, 4) is None, h
    assert anchored_letter("Answer: C, because A is wrong", 4) == "C" and anchored_letter("Answer: C. A is wrong", 4) == "C"
    # "answer" needs a colon or "is": a closing remark about 'the answer I gave' is not an answer (I is a valid letter of a 10-option question)
    assert anchored_letter("Answer: C\n\nThe answer I gave above is final.", 10) == "C" and anchored_letter("Answer C", 4) is None and anchored_letter("Answer: B\nAnswer: C", 4) == "C"


def test_unbiased_pass_at_k():
    assert unbiased_pass_at_k(10, 0, 3) == 0.0 and unbiased_pass_at_k(10, 10, 3) == 1.0
    assert abs(unbiased_pass_at_k(10, 3, 1) - 0.3) < 1e-12 and unbiased_pass_at_k(4, 3, 2) == 1.0


# ---- prompts -----------------------------------------------------------------------------------------------------------------

def test_mc_prompt_letters_follow_the_option_count():
    q = chat_evals.mc_query("Which?", ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    assert "where LETTER is one of ABCDEFGHIJ." in q and q.endswith("J) j") and "\n\nWhich?\n\nA) a\n" in q
    assert "one of ABCD." in chat_evals.mc_query("Q?", ["a", "b", "c", "d"])


def test_prompts_have_no_instruction_and_a_fixed_gpqa_order():
    line = {"Question": "Q text", "Correct Answer": "right", "Incorrect Answer 1": "w1", "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3",
            "High-level domain": "Physics", "Subdomain": "x", "Record ID": "rec1"}
    d1, d2 = chat_evals.gpqa_chat_prompt(line, "t"), chat_evals.gpqa_chat_prompt(line, "t")
    assert d1.query == d2.query and d1.gold_index == d2.gold_index, "no random.randint: the same order in every run"
    assert d1.instruction is None and d1.choices == ["A", "B", "C", "D"]
    assert f"{'ABCD'[d1.gold_index]}) right" in d1.query
    mp = chat_evals.mmlu_pro_chat_prompt({"question": "Q", "options": [str(i) for i in range(9)], "answer_index": 3, "category": "law", "question_id": 1}, "t")
    assert mp.choices == list("ABCDEFGHI") and mp.gold_index == 3 and "one of ABCDEFGHI." in mp.query
    ifd = chat_evals.ifeval_chat_prompt({"prompt": "  Write.  ", "instruction_id_list": ["a:b"], "kwargs": [{}]}, "t")
    assert ifd.query == "  Write.  " and ifd.instruction is None and ifd.specific["instructions_id_list"] == ["a:b"]
    ibd = chat_evals.ifbench_chat_prompt({"prompt": "P", "instruction_id_list": ["a:b"], "kwargs": [{}]}, "t")
    assert ibd.specific["instruction_id_list"] == ["a:b"]
    md = chat_evals.math_chat_prompt({"problem": "  1+1?  ", "answer": "2"}, "t", gold_of=lambda l: l["answer"])
    assert md.query.endswith("\n\n1+1?") and "\\boxed{ANSWER}" in md.query and "Problem:" not in md.query and md.choices == ["2"]
    assert chat_evals.mmlu_redux_chat_prompt({"question": "Q", "choices": ["a", "b", "c", "d"], "answer": 1, "error_type": "wrong_groundtruth"}, "t", subject="x") is None


def test_if_metrics_are_the_base_checkers_made_deterministic():
    import langdetect
    from lighteval.tasks.registry import Registry
    ts = Registry(tasks="ifeval_chat|0,ifbench_chat|0").load_tasks()
    names = ["prompt_level_strict_acc", "inst_level_strict_acc", "prompt_level_loose_acc", "inst_level_loose_acc"]
    for name, line_fn, line in (("ifeval_chat|0", chat_evals.ifeval_chat_prompt, {"prompt": "Write.", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}),
                                ("ifbench_chat|0", chat_evals.ifbench_chat_prompt, {"prompt": "Say hi.", "instruction_id_list": ["count:keywords_multiple"],
                                                                                   "kwargs": [{"keyword1": "hi", "keyword2": "a", "keyword3": "b", "keyword4": "c", "keyword5": "d"}]})):
        m = ts[name].metrics[0]
        assert m.metric_name == names + ["checker_error"] and isinstance(m.sample_level_fn, chat_evals._DeterministicChecks) and m.category.name == "GENERATIVE"
        langdetect.DetectorFactory.seed = None
        out = m.sample_level_fn.compute(line_fn(line, name), _resp(""))
        assert out["prompt_level_strict_acc"] == 0 and out["prompt_level_loose_acc"] == 0 and out["inst_level_strict_acc"] == [False], "an empty answer follows nothing"
        assert langdetect.DetectorFactory.seed == 0, "langdetect is seeded before every sample"
        assert out["checker_error"] == 0 and "checker_error" in m.metric_name and m.higher_is_better["checker_error"] is False
        # a checker that raises scores 0 on every instruction and is counted (it must not stop the run)
        wrapped = chat_evals._DeterministicChecks(type("Boom", (), {"compute": lambda self, d, r, **k: (_ for _ in ()).throw(IndexError("odd answer"))})())
        d = line_fn(line, name)
        boom = wrapped.compute(d, _resp("odd"))
        assert boom["checker_error"] == 1 and boom["prompt_level_strict_acc"] == 0 and boom["inst_level_loose_acc"] == [False]
    import random
    random.seed(5)
    before = random.getstate()
    chat_evals._DeterministicChecks(type("Id", (), {"compute": lambda self, d, r, **k: {}})()).compute(Doc(query="q", choices=[""], gold_index=0, task_name="t"), _resp("x"))
    assert random.getstate() == before, "the global random state is restored"


def test_default_sample_counts():
    from lighteval.tasks.registry import Registry
    tasks = Registry(tasks="aime24_chat|0,aime25_chat|0,aime26_chat|0,hmmt_feb_2026_chat|0,math_500_chat|0,gpqa_diamond_chat|0,mmlu_pro_chat|0,ifeval_chat|0,ifbench_chat|0").load_tasks()
    got = {k.split("|")[0]: max(t.num_samples) for k, t in tasks.items()}
    assert got == {"aime24_chat": 32, "aime25_chat": 32, "aime26_chat": 32, "hmmt_feb_2026_chat": 32, "math_500_chat": 4, "gpqa_diamond_chat": 8,
                   "mmlu_pro_chat": 1, "ifeval_chat": 1, "ifbench_chat": 1}, got
    assert {t.generation_size for t in tasks.values()} == {32768} and all(not t.stop_sequence for t in tasks.values())


# ---- model config fields and the num_samples override -----------------------------------------------------------------------

def test_vllm_config_has_the_chat_fields_and_defaults_are_off(tmp_path=None):
    import tempfile
    import yaml

    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    d = VLLMModelConfig(model_name="m")
    assert d.chat_template is None and d.num_samples is None and d.error_on_prompt_truncation is False
    with tempfile.TemporaryDirectory() as t:
        f = f"{t}/m.yaml"
        open(f, "w").write(yaml.safe_dump({"model_parameters": {"model_name": "m", "override_chat_template": True, "chat_template_kwargs": {"enable_thinking": False},
                                                                  "chat_template": "/x.jinja", "num_samples": 8, "error_on_prompt_truncation": True,
                                                                  "generation_parameters": {"temperature": 0.6, "top_k": 20, "max_new_tokens": 100}}}))
        c = VLLMModelConfig.from_path(f)
    assert c.chat_template_kwargs == {"enable_thinking": False} and c.num_samples == 8 and c.generation_parameters.top_k == 20


class _Tok:
    chat_template = "x"

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [list(range(len(t))) for t in texts]}


def _stub_model(cfg, captured):
    from lighteval.models.vllm.vllm_model import VLLMModel

    class Stub(VLLMModel):
        def __init__(self):
            self.config = cfg
            self.use_chat_template = True
            self._tokenizer = _Tok()
            self._max_length = 100
            self._add_special_tokens = False

            class PM:
                @staticmethod
                def prepare_prompt(doc):
                    return doc.query

            self.prompt_manager = PM()

        def _generate(self, inputs, max_new_tokens=None, stop_tokens=None, returns_logits=False, num_samples=1, **kw):
            captured["n"] = num_samples
            from types import SimpleNamespace as NS
            return [NS(outputs=[NS(token_ids=[1, 2], text="t", finish_reason="stop")] * num_samples, prompt_token_ids=i) for i in inputs]

    return Stub()


def _doc(n, query="abcdef"):
    d = Doc(query=query, choices=["a"], gold_index=0, task_name="t")
    d.num_samples = n
    d.generation_size = 10
    return d


def test_num_samples_override_only_for_multi_sample_tasks():
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    cap = {}
    m = _stub_model(VLLMModelConfig(model_name="m", num_samples=5), cap)
    m._greedy_until([_doc(32)])
    assert cap["n"] == 5, "a task that samples more than once takes the run's n"
    m._greedy_until([_doc(1)])
    assert cap["n"] == 1, "an n = 1 task keeps one sample"
    m = _stub_model(VLLMModelConfig(model_name="m"), cap)
    m._greedy_until([_doc(32)])
    assert cap["n"] == 32, "no override: the task's own n"


def test_prompt_truncation_is_an_error_when_asked():
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    long_doc = _doc(1, query="x" * 95)                      # 95 prompt tokens + 10 new > max length 100
    _stub_model(VLLMModelConfig(model_name="m"), {})._greedy_until([long_doc])              # upstream: cut from the left, with a warning
    try:
        _stub_model(VLLMModelConfig(model_name="m", error_on_prompt_truncation=True), {})._greedy_until([long_doc])
    except ValueError as e:
        assert "chat template head" in str(e)
    else:
        raise AssertionError("chat runs must refuse to cut a prompt")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok  {name}")
            except unittest.SkipTest as e:
                print(f"skip {name}: {e}")
