# MIT License
#
# titan_eval (owner 11-16): the exact prompt count of the long-context chat tasks with `over_context: zero` (exact_prompt.py): the prompt is rendered as the vLLM model renders it, counted with the
# model's tokenizer, margin 0 (the OLMES rule), and the vLLM model checks the count against the token ids it sends and the engine returns.

import contextlib
import json
import os
from types import SimpleNamespace
from unittest import mock

from lighteval.models.vllm.vllm_model import VLLMModel
from lighteval.tasks import answer_fit as af, exact_prompt
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc


class FakeChatTok:
    """A chat tokenizer without a model. The template writes a marker around every message, the generation prompt `<|assistant|>` and, with `enable_thinking=False`, an empty think block;
    every whitespace-separated piece is one token. `calls` = the batches `__call__` got."""

    chat_template = "x"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        assert tokenize is False and add_generation_prompt is True
        think = "<think> </think> " if kw.get("enable_thinking") is False else ""
        return "".join(f"<|{m['role']}|> {m['content']} <|end|> " for m in messages) + "<|assistant|> " + think

    def __call__(self, texts, add_special_tokens=False):
        assert isinstance(texts, list)
        self.calls.append((list(texts), add_special_tokens))
        return {"input_ids": [t.split() for t in texts]}


ENV = {"TITAN_EVAL_TOKENIZER": "fake", "TITAN_EVAL_MAX_MODEL_LEN": "1000", "TITAN_EVAL_RENDER": ""}


@contextlib.contextmanager
def fake(tok=None, **render):
    """A run that exports its tokenizer (and render settings): the loader is replaced by one that returns `tok`."""
    exact_prompt.reset()
    env = dict(ENV, TITAN_EVAL_RENDER=json.dumps(render) if render else "")
    with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(exact_prompt, "load_tokenizer", return_value=tok or FakeChatTok()):
        yield
    exact_prompt.reset()


def test_no_tokenizer_in_the_environment_means_no_exact_count():
    with mock.patch.dict(os.environ, {"TITAN_EVAL_TOKENIZER": ""}, clear=False):
        assert exact_prompt.get() is None and exact_prompt.render_settings() is None


def test_settings_default_to_a_plain_chat_run_and_come_from_the_render_variable():
    with mock.patch.dict(os.environ, {"TITAN_EVAL_TOKENIZER": "/m", "TITAN_EVAL_RENDER": ""}, clear=False):
        s = exact_prompt.render_settings()
    assert s == {"chat_template": None, "chat_template_kwargs": {}, "system_prompt": None, "add_special_tokens": False, "trust_remote_code": False, "tokenizer": "/m"}
    raw = '{"chat_template": "/t.jinja", "chat_template_kwargs": {"enable_thinking": false}, "system_prompt": "S", "add_special_tokens": null, "trust_remote_code": true, "other": 1}'
    with mock.patch.dict(os.environ, {"TITAN_EVAL_TOKENIZER": "/m", "TITAN_EVAL_RENDER": raw}, clear=False):
        s = exact_prompt.render_settings()
    assert s["chat_template"] == "/t.jinja" and s["chat_template_kwargs"] == {"enable_thinking": False} and s["system_prompt"] == "S" and s["add_special_tokens"] is False and s["trust_remote_code"] is True
    assert "other" not in s


def test_render_is_the_prompt_managers_render_with_the_runs_settings():
    """The thinking switch, the system prompt, a doc's own conversation and a plain-text doc are rendered by the SAME PromptManager call as the vLLM model's."""
    tok = FakeChatTok()
    with fake(tok, chat_template_kwargs={"enable_thinking": False}, system_prompt="SYS"):
        ex = exact_prompt.get()
        user = Doc(query="a b c", choices=[""], gold_index=0)
        conv = Doc(query="x", choices=[""], gold_index=0, specific={"chat_messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]})
        plain = Doc(query="raw text here", choices=[""], gold_index=0, specific={"plain_text": True})
        ref = PromptManager(True, tok, "SYS", chat_template_kwargs={"enable_thinking": False})
        for d in (user, conv, plain):
            assert ex.render(d) == ref.prepare_prompt(d)
        assert ex.render(user) == "<|system|> SYS <|end|> <|user|> a b c <|end|> <|assistant|> <think> </think> "
        assert ex.render(plain) == "raw text here"
        assert ex.count(user) == len(ref.prepare_prompt(user).split()), "the count is the tokens of the whole rendered string"
    with fake(tok):                                                                 # thinking left to the template: no empty block, no system message
        assert exact_prompt.get().render(Doc(query="a b c", choices=[""], gold_index=0)) == "<|user|> a b c <|end|> <|assistant|> "


def test_the_count_is_of_the_whole_rendered_string_with_the_runs_add_special_tokens():
    tok = FakeChatTok()
    with fake(tok, add_special_tokens=True):
        ex = exact_prompt.get()
        n = ex.count(Doc(query="w " * 10, choices=[""], gold_index=0))
    assert n == 10 + 3 and tok.calls[-1][1] is True, "10 words + the 3 template tokens (<|user|>, <|end|>, <|assistant|>); add_special_tokens as the run's model config has it"
    with fake(tok):
        exact_prompt.get().count(Doc(query="w", choices=[""], gold_index=0))
    assert tok.calls[-1][1] is False, "chat default: nothing added (no BOS)"


def test_counts_are_cached_per_rendered_string_and_prefetch_counts_in_batches():
    tok = FakeChatTok()
    rows = [{"input": f"row {i} " + "w " * i} for i in range(1, 6)]
    docs_of = lambda row, name: [Doc(query=row["input"], choices=[""], gold_index=0)]                  # noqa: E731
    with fake(tok):
        exact_prompt.prefetcher(docs_of)(rows, "t")
        assert len(tok.calls) == 1 and len(tok.calls[0][0]) == 5, "one batch call for all five rows"
        ex = exact_prompt.get()
        counts = [ex.count(docs_of(r, "t")[0]) for r in rows]
        assert len(tok.calls) == 1, "the per-row calls find their count in the cache"
        assert counts == [len(ex.render(docs_of(r, "t")[0]).split()) for r in rows]
        with mock.patch.object(exact_prompt, "COUNT_CHUNK_CHARS", 10):
            tok.calls.clear()
            ex.prefetch(Doc(query=f"another {i} " * 3, choices=[""], gold_index=0) for i in range(4))
            assert len(tok.calls) == 4, "a chunk is closed once its characters reach the limit"
    with mock.patch.dict(os.environ, {"TITAN_EVAL_TOKENIZER": ""}, clear=False):
        exact_prompt.prefetcher(docs_of)(rows, "t")                                                       # no tokenizer: nothing happens
    with fake(tok):
        tok.calls.clear()
        exact_prompt.prefetcher(docs_of, lambda name: False)(rows, "t")
        assert tok.calls == [], "an inactive task is not prefetched"


def test_stamp_writes_the_count_under_the_keys_the_vllm_model_and_the_details_use():
    assert exact_prompt.stamp({"a": 1}, 7) == {"a": 1, "n_tokens": 7, "exact_prompt_tokens": 7}


def test_fit_has_no_margin():
    """OLMES eleuther_vllm_causallms.py:213: budget = min(task budget, context - prompt), the item is dropped only below the floor."""
    assert not hasattr(af, "MARGIN") and not hasattr(af, "CHAT_TEMPLATE_TOKENS")
    f = af.fit_answer(900, 100, 100, ctx=1000)
    assert (f["over_context"], f["answer_budget"], f["reduced_budget"]) == (False, 100, False), "prompt + answer = the window exactly: fits"
    f = af.fit_answer(901, 100, 100, ctx=1000)
    assert (f["over_context"], f["answer_budget"]) == (True, 0)
    f = af.fit_answer(500, 1000, 100, ctx=1000)
    assert (f["over_context"], f["answer_budget"], f["reduced_budget"]) == (False, 500, True)
    with mock.patch.dict(os.environ, {"TITAN_EVAL_MAX_MODEL_LEN": ""}, clear=False):
        assert af.fit_answer(10, 100, 50) == af.no_fit(100), "no window known: nothing is marked"


# ---- the vLLM model checks the count ------------------------------------------------------------------------------------------------

def _doc(n_exact, skip=False):
    d = Doc(query="q", choices=[""], gold_index=0)
    d.id = "7"
    d.specific = {"exact_prompt_tokens": n_exact, **({"no_generation": True} if skip else {})} if n_exact is not None else {}
    return d


def test_the_vllm_model_fails_loudly_on_a_count_that_differs_and_ignores_the_rest():
    check = VLLMModel._check_exact_counts
    check([_doc(5), _doc(6), _doc(None), _doc(9, skip=True)], [5, 6, 99, 1], [False, False, False, True], "x")           # equal / no count / not generated: fine
    try:
        check([_doc(5), _doc(6)], [5, 7], [False, False], "the prompts about to be sent")
    except RuntimeError as e:
        assert "1 of 2 docs" in str(e) and "counted 6, vLLM side 7" in str(e) and "the prompts about to be sent" in str(e)
    else:
        raise AssertionError("a different count must raise")
    check([_doc(5)], [None], [False], "x")                                                                                 # not generated (no output): nothing to compare


def test_the_engine_reported_prompt_length_wins_over_the_ids_we_sent():
    out = SimpleNamespace(prompt_token_ids=[1, 2, 3])
    assert VLLMModel._engine_prompt_len(out) == 3
    out.engine_prompt_len = 5                                                  # a data-parallel worker: the length of the ids the ENGINE returned
    assert VLLMModel._engine_prompt_len(out) == 5


def test_the_dp_worker_slim_output_carries_the_engine_prompt_length_and_old_tuples_still_load():
    from lighteval.models.vllm.dp_pool import rebuild_output, slim_output

    req = SimpleNamespace(outputs=[SimpleNamespace(text="t", token_ids=[1], finish_reason="stop")], prompt_logprobs=None, prompt_token_ids=[4, 5, 6, 7])
    slim = slim_output(req)
    assert slim[2] == 4
    assert rebuild_output(slim, [9, 9]).engine_prompt_len == 4
    assert rebuild_output(slim[:2], [9, 9]).engine_prompt_len is None, "a two-element tuple (an engine that does not report it)"


def test_the_task_hands_the_rows_of_a_split_to_the_prefetch_hook_of_its_prompt_function():
    """LightevalTask._get_docs_from_split: a prompt function with a `prefetch` attribute gets the rows first (one batch call), the per-row calls then hit the cache; few-shot pools are not prefetched."""
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.tasks import ruler

    rows = [{"input": "w " * i + "Q?", "outputs": ["x"], "gen_prefix": "A:", "length": 10} for i in (2, 4, 6)]
    name = "ruler_4096_chat:qa_1|0"
    calls = []

    class Tok(FakeChatTok):
        def __call__(self, texts, add_special_tokens=False):
            calls.append(len(texts))
            return super().__call__(texts, add_special_tokens)

    this = SimpleNamespace(dataset={"train": rows}, formatter=ruler.ruler_chat_prompt, name=name, generation_grammar=None, generation_size=50, stop_sequence=[])
    with fake(Tok()):
        docs = LightevalTask._get_docs_from_split(this, ["train"])
        assert calls == [3], "ONE batch call for the three rows (the per-row calls found their counts in the cache)"
        assert [d.specific["n_tokens"] for d in docs] == [2 + 1 + 3, 4 + 1 + 3, 6 + 1 + 3]
        calls.clear()
        exact_prompt.get()._counts.clear()
        LightevalTask._get_docs_from_split(this, ["train"], few_shots=True)
        assert calls == [1, 1, 1], "a few-shot pool is not prefetched: the rows are counted by the per-row calls, one by one"


def test_a_fast_tokenizer_is_counted_through_its_backend_and_a_difference_falls_back_to_the_tokenizer_call():
    """encode_batch_fast gives the same ids without the offsets (4x faster on 100k-token prompts); the first batch is probed against the tokenizer call, a difference switches to that call."""

    class Enc(list):
        pass

    class Backend:
        def __init__(self, off=0):
            self.off, self.calls = off, 0

        def encode_batch_fast(self, texts, add_special_tokens=False):
            self.calls += 1
            return [Enc(t.split()) if not self.off else Enc(t.split() + ["x"] * self.off) for t in texts]

    class Tok(FakeChatTok):
        def __init__(self, off=0):
            super().__init__()
            self.backend_tokenizer = Backend(off)

    texts = ["a b c", "d e", "f g h i j"]
    good = Tok()
    with fake(good):
        ex = exact_prompt.get()
        assert ex.count_texts(texts) == [3, 2, 5] and good.backend_tokenizer.calls == 1 and ex._fast_ok is True
        assert len(good.calls) == 1 and len(good.calls[0][0]) == 2, "the probe: the tokenizer call on the two shortest strings only"
        ex.count_texts(["k l"])
        assert len(good.calls) == 1, "probed once per run"
    bad = Tok(off=1)                                                       # the fast path counts one token too many: the run falls back to the tokenizer call, with the right counts
    with fake(bad):
        ex = exact_prompt.get()
        assert ex.count_texts(texts) == [3, 2, 5] and ex._fast_ok is False
        n = len(bad.calls)
        assert ex.count_texts(["m n o p"]) == [4] and len(bad.calls) == n + 1 and bad.backend_tokenizer.calls == 1, "after the fallback the fast path is never used"
