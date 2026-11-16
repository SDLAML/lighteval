# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from lighteval.tasks.tasks import ruler


def test_ruler_prompt_appends_generation_prefix():
    doc = ruler.ruler_prompt(
        {
            "input": "Question text",
            "gen_prefix": "Answer:",
            "outputs": ["Paris"],
        },
        "ruler_512:qa_1|0",
    )

    assert doc.query == "Question text\nAnswer:"
    assert doc.choices == ["Paris"]
    assert doc.gold_index == [0]
    assert doc.task_name == "ruler_512:qa_1|0"


def test_runtime_prompt_helper_matches_ruler_prompt():
    line = {
        "input": "Question text",
        "gen_prefix": "Answer:",
        "outputs": ["Paris"],
    }
    doc = ruler.ruler_prompt(line, "ruler_512:qa_1|0")

    assert ruler._build_runtime_prompt(line["input"], line["gen_prefix"]) == doc.query


def test_ruler_generation_sizes_match_reference_defaults():
    tasks = ruler.get_ruler_tasks(
        "dummy-tokenizer",
        lengths=[2048],
        subsets=["niah_single_1", "vt", "cwe", "fwe", "qa_2"],
    )
    task_map = {task.name: task.generation_size for task in tasks}

    assert task_map["ruler_2048:niah_single_1"] == 50
    assert task_map["ruler_2048:vt"] == 50
    assert task_map["ruler_2048:cwe"] == 100
    assert task_map["ruler_2048:fwe"] == 50
    assert task_map["ruler_2048:qa_2"] == 50


def test_ruler_qa_prompt_matches_reference_template():
    prompt, answers = ruler._qa_generate_input_output(
        0,
        1,
        qas=[{"query": "Where is the Eiffel Tower?", "outputs": ["Paris"], "context": [0]}],
        docs=["Paris is the capital of France."],
    )

    assert prompt == (
        "Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given documents.\n\n"
        "Document 1:\nParis is the capital of France.\n\n"
        "Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\n"
        "Question: Where is the Eiffel Tower?"
    )
    assert answers == ["Paris"]


def test_ruler_uses_singular_niah_template_for_single_tasks(monkeypatch):
    captured = {}

    def fake_niah_generate_samples(*args, **kwargs):
        captured["template"] = kwargs["template"]
        return []

    monkeypatch.setattr(ruler, "_niah_generate_samples", fake_niah_generate_samples)

    ruler._generate_subset("niah_single_1", 128, tokenizer=None)

    assert captured["template"] == ruler.NIAH_TEMPLATE_SINGLE


def test_ruler_uses_plural_niah_template_for_multi_tasks(monkeypatch):
    captured = {}

    def fake_niah_generate_samples(*args, **kwargs):
        captured["template"] = kwargs["template"]
        return []

    monkeypatch.setattr(ruler, "_niah_generate_samples", fake_niah_generate_samples)
    monkeypatch.setattr(ruler, "_ensure_nltk", lambda: None)
    monkeypatch.setattr(ruler, "_get_essay_haystack", lambda: ["one", "two"])

    ruler._generate_subset("niah_multivalue", 128, tokenizer=None)

    assert captured["template"] == ruler.NIAH_TEMPLATE_MULTI


def test_ruler_budget_helper_uses_newline_before_generation_prefix():
    prompt = ruler._build_runtime_prompt("Question text", "Answer:")

    assert prompt == "Question text\nAnswer:"


class _FakeTokenizer:
    def __call__(self, text):
        return type("Tokens", (), {"input_ids": list(range(len(text)))})()


def test_qa_length_matches_runtime_prompt_shape():
    input_text = "Question text"
    gen_prefix = "Answer:"
    tokens_to_generate = ruler.QA_CONFIG["tokens_to_generate"]

    expected = len(_FakeTokenizer()(ruler._build_runtime_prompt(input_text, gen_prefix)).input_ids) + tokens_to_generate

    assert expected == len("Question text\nAnswer:") + tokens_to_generate


def test_cwe_full_prompt_length_includes_icl_example():
    full_input = "Example block\nTask text"
    gen_prefix = "Answer: The top 10 words that appear most often in the list are:"
    tokens_to_generate = ruler.CWE_CONFIG["tokens_to_generate"]

    expected = len(_FakeTokenizer()(ruler._build_runtime_prompt(full_input, gen_prefix)).input_ids) + tokens_to_generate

    assert expected == len(full_input + "\n" + gen_prefix) + tokens_to_generate


def test_vt_cached_sample_uses_final_runtime_prompt_shape():
    task_prompt = ruler.VT_TEMPLATE.format(
        context="VAR A = 12345",
        query="12345",
        num_v=5,
    )
    cached_input, gen_prefix = ruler._vt_build_cached_sample(task_prompt, "ICL EXAMPLE")
    length = ruler._runtime_prompt_budget_length(
        _FakeTokenizer(),
        cached_input,
        gen_prefix,
        ruler.VT_CONFIG["tokens_to_generate"],
    )

    assert cached_input == (
        "ICL EXAMPLE\n\n"
        "Memorize and track the chain(s) of variable assignment hidden in the "
        "following text.\n\nVAR A = 12345\n"
        "Question: Find all variables that are assigned the value 12345 in the text above."
    )
    assert gen_prefix == (
        "Answer: According to the chain(s) of variable assignment in the text above, "
        "5 variables are assigned the value 12345, they are:"
    )
    assert length == len(cached_input + "\n" + gen_prefix) + ruler.VT_CONFIG["tokens_to_generate"]


def test_fwe_length_matches_runtime_prompt_shape():
    full_prompt = (
        "Read the following coded text and track the frequency of each coded word. "
        "Find the three most frequently appeared coded words.\nalpha beta gamma\n"
        "Question: Do not provide any explanation. Please ignore the dots '....'. "
        "What are the three most frequently appeared words in the above coded text?"
        " Answer: According to the coded text above, the three most frequently appeared words are:"
    )
    input_text, gen_prefix = ruler._fwe_build_cached_sample(full_prompt)
    length = ruler._runtime_prompt_budget_length(
        _FakeTokenizer(),
        input_text,
        gen_prefix,
        ruler.FWE_CONFIG["tokens_to_generate"],
    )

    assert length == len(input_text + "\n" + gen_prefix) + ruler.FWE_CONFIG["tokens_to_generate"]


def test_non_qa_cached_lengths_match_retokenized_runtime_prompts():
    samples = [
        {
            "input": "ICL EXAMPLE\n\nTask block",
            "gen_prefix": "Answer: According to the chain(s) of variable assignment in the text above, 5 variables are assigned the value 12345, they are:",
            "tokens_to_generate": ruler.VT_CONFIG["tokens_to_generate"],
            "max_length": 500,
        },
        {
            "input": "Example block\nTask text",
            "gen_prefix": "Answer: The top 10 words that appear most often in the list are:",
            "tokens_to_generate": ruler.CWE_CONFIG["tokens_to_generate"],
            "max_length": 500,
        },
        {
            "input": "Task text",
            "gen_prefix": "Answer: According to the coded text above, the three most frequently appeared words are:",
            "tokens_to_generate": ruler.FWE_CONFIG["tokens_to_generate"],
            "max_length": 500,
        },
    ]

    for sample in samples:
        length = ruler._runtime_prompt_budget_length(
            _FakeTokenizer(),
            sample["input"],
            sample["gen_prefix"],
            sample["tokens_to_generate"],
        )

        assert length == len(sample["input"] + "\n" + sample["gen_prefix"]) + sample["tokens_to_generate"]
        assert length <= sample["max_length"]


# ---- chat variant (titan_eval, owner 11-16): ruler_<L>_chat:<subset> ---------------------------------------------------------------------------------------------------------------


def test_ruler_chat_prompt_is_the_prepared_input_without_the_answer_prefix():
    line = {
        "input": "Some text.\nWhat is the special magic number for tenuous-thirst mentioned in the provided text?",
        "gen_prefix": "The special magic number for tenuous-thirst mentioned in the provided text is",
        "outputs": ["2867825"],
    }
    doc = ruler.ruler_chat_prompt(line, "ruler_4096_chat:niah_single_1|0")

    assert doc.query == line["input"], "the user message is the RULER prompt as prepared (its question is the last line)"
    assert line["gen_prefix"] not in doc.query
    assert doc.choices == ["2867825"] and doc.gold_index == [0] and doc.task_name == "ruler_4096_chat:niah_single_1|0"
    assert not doc.specific, "no gen_prefix in specific: the scorer must not prepend the answer prefix to a chat reply"
    base = ruler.ruler_prompt(line, "ruler_4096:niah_single_1|0")
    assert base.query == line["input"] + "\n" + line["gen_prefix"] and base.specific == {"gen_prefix": line["gen_prefix"]}, "the base task is unchanged"


def test_ruler_chat_tasks_share_data_budgets_and_scorer_with_the_base_tasks():
    subsets = ["niah_single_1", "niah_multivalue", "vt", "cwe", "fwe", "qa_1", "qa_2"]
    tasks = {t.name: t for t in ruler.get_ruler_tasks("dummy-tokenizer", lengths=[4096, 8192], subsets=subsets)}
    assert len(tasks) == 2 * 2 * len(subsets), "base and chat variant of every (length, subset)"
    for length in (4096, 8192):
        for sub in subsets:
            base, chat = tasks[f"ruler_{length}:{sub}"], tasks[f"ruler_{length}_chat:{sub}"]
            assert chat.hf_repo == base.hf_repo, "the SAME prepared folder: nothing extra to prepare"
            assert base.generation_size == ruler._gen_size(sub, length) and chat.generation_size == ruler._chat_gen_size(sub, length) >= base.generation_size, "own size (base); chat: max(own, NVIDIA tokens_to_generate), vt 100 (review 11-16 R1)"
            assert [m.metric_name for m in base.metrics] == ["ruler_match"] and [m.metric_name for m in chat.metrics] == [["ruler_match", "over_context"]], "the base scorer + the share not generated"
            assert base.prompt_function is ruler.ruler_prompt and chat.prompt_function is ruler.ruler_chat_prompt
            assert not chat.stop_sequence and not base.stop_sequence, "no stop sequence: RULER has none, a chat reply ends at the end of turn"
    assert tasks["ruler_4096_chat:niah_multivalue"].generation_size == 300 and tasks["ruler_8192_chat:niah_multivalue"].generation_size == 250
    assert tasks["ruler_8192_chat:cwe"].generation_size == 120 and tasks["ruler_8192_chat:vt"].generation_size == 100 and tasks["ruler_8192:vt"].generation_size == 50


def test_ruler_registers_the_256k_length_with_the_sizes_of_the_other_lengths():
    """Owner 11-16: RULER 256k (262,144 tokens, base and chat twin). The default table holds it; its sizes are the 128k ones (niah_multivalue 250 above 4k)."""
    assert ruler.DEFAULT_LENGTHS[-1] == 262144 and ruler.DEFAULT_LENGTHS == sorted(set(ruler.DEFAULT_LENGTHS))
    tasks = {t.name: t for t in ruler.get_ruler_tasks("dummy-tokenizer")}
    assert len(tasks) == 2 * len(ruler.DEFAULT_LENGTHS) * len(ruler.SUBSETS)
    for sub in ruler.SUBSETS:
        base, chat = tasks[f"ruler_262144:{sub}"], tasks[f"ruler_262144_chat:{sub}"]
        assert base.generation_size == ruler._gen_size(sub, 131072) == ruler._gen_size(sub, 262144)
        assert chat.generation_size == ruler._chat_gen_size(sub, 131072) == ruler._chat_gen_size(sub, 262144)
        assert chat.hf_repo == base.hf_repo and base.hf_repo.endswith(f"/262144/{sub}"), "the chat twin reads the base folder"
    assert tasks["ruler_262144:niah_multivalue"].generation_size == 250
    assert ruler.prepared_status("dummy-tokenizer", [262144], ["vt"])["missing"] == [(262144, "vt")], "262144 is a valid generated length (>= 2048)"


def test_ruler_chat_scoring_is_ruler_match_on_the_reply_only():
    from lighteval.metrics.metrics_sample import RulerStringMatch
    from lighteval.models.model_output import ModelResponse

    metric = RulerStringMatch()
    line = {"input": "Q?", "gen_prefix": "Answer: The top 10 words that appear most often in the list are:", "outputs": ["alpha", "beta", "list"]}
    chat = ruler.ruler_chat_prompt(line, "ruler_4096_chat:cwe|0")
    base = ruler.ruler_prompt(line, "ruler_4096:cwe|0")
    reply = ModelResponse(text=["1. alpha 2. beta"])
    assert metric.compute(chat, reply) == 2 / 3, "a fraction of the golds found in the reply"
    assert metric.compute(base, reply) == 1.0, "the base scorer prepends the answer prefix (it contains the gold word `list`): the chat doc must not get that credit"
    # a thinking reply: the scored text is what follows the last </think> (final_text of the strict-reasoning run)
    think = ModelResponse(text=["<think>beta alpha</think> nothing"])
    think.text_post_processed = ["nothing"]
    assert metric.compute(chat, think) == 0.0, "the scorer reads final_text, never the think block"
    qa = ruler.ruler_chat_prompt({"input": "Q?", "gen_prefix": "Answer:", "outputs": ["Paris", "paris, france"]}, "ruler_4096_chat:qa_1|0")
    assert metric.compute(qa, ModelResponse(text=["It is Paris."])) == 1.0, "QA: any gold, normalised substring (the `:qa_` rule holds for the chat names)"
    assert metric.compute(qa, ModelResponse(text=["London"])) == 0.0


def _exact_env(ctx, **render):
    """A chat run of a fake chat tokenizer (the template writes <|user|> .. <|end|> <|assistant|> [<think> </think>], one token per whitespace-separated piece) that exports its window and render settings."""
    import contextlib
    import json
    import os
    from unittest import mock

    from lighteval.tasks import exact_prompt

    class Tok:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            think = "<think> </think> " if kw.get("enable_thinking") is False else ""
            return "".join(f"<|{m['role']}|> {m['content']} <|end|> " for m in messages) + "<|assistant|> " + think

        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [t.split() for t in texts]}

    @contextlib.contextmanager
    def cm():
        exact_prompt.reset()
        env = {"TITAN_EVAL_TOKENIZER": "fake", "TITAN_EVAL_MAX_MODEL_LEN": str(ctx), "TITAN_EVAL_RENDER": json.dumps(render) if render else ""}
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(exact_prompt, "load_tokenizer", return_value=Tok()):
            yield
        exact_prompt.reset()

    return cm()


def test_ruler_chat_item_that_does_not_fit_the_window_is_marked_and_scores_zero():
    """Owner 11-16 (`over_context: zero`, the OLMES rule, margin 0): floor = the subtask's own answer size; the chat prompt is rendered as the vLLM model sends it and counted exactly."""
    from lighteval.models.model_output import ModelResponse

    # qa_1 at 4096: own answer size 50; the answer prefix is NOT in the chat prompt. 3,946 words + <|user|> <|end|> <|assistant|> = 3,949 tokens
    line = {"input": "w " * 3946 + "Q?", "gen_prefix": "Answer:", "outputs": ["Paris"], "length": 4000}
    name = "ruler_4096_chat:qa_1|0"
    n_exact = 3947 + 3
    with _exact_env(n_exact + 50) as _:
        ok = ruler.ruler_chat_prompt(line, name)
    assert ok.specific["over_context"] is False and ok.specific["n_tokens"] == n_exact == ok.specific["exact_prompt_tokens"] and ok.specific["answer_budget"] == 50, "prompt + answer = the window exactly: fits (no margin)"
    assert "no_generation" not in ok.specific and ok.query == line["input"]
    with _exact_env(n_exact + 49) as _:
        over = ruler.ruler_chat_prompt(line, name)
    assert over.specific["over_context"] and over.specific["no_generation"] and over.query == "Reply with the word OK." and over.specific["answer_budget"] == 0
    assert over.specific["exact_prompt_tokens"] == n_exact, "the count of the real prompt stays in the record"
    with _exact_env(n_exact + 52, chat_template_kwargs={"enable_thinking": False}) as _:           # the thinking switch is part of the prompt: the empty think block adds 2 tokens
        off = ruler.ruler_chat_prompt(line, name)
    assert off.specific["n_tokens"] == n_exact + 2 and off.specific["over_context"] is False
    with _exact_env(n_exact + 51, chat_template_kwargs={"enable_thinking": False}) as _:
        assert ruler.ruler_chat_prompt(line, name).specific["over_context"] is True
    with _exact_env(n_exact + 50, system_prompt="be brief") as _:                                   # a system prompt is rendered too: 2 words + 2 markers more
        assert ruler.ruler_chat_prompt(line, name).specific["n_tokens"] == n_exact + 4
    metric = ruler.ruler_chat_metric.sample_level_fn
    assert metric.compute(over, ModelResponse(text=["Paris"])) == {"ruler_match": 0.0, "over_context": 1.0}, "an item not generated scores 0 whatever came back"
    assert metric.compute(ok, ModelResponse(text=["It is Paris."])) == {"ruler_match": 1.0, "over_context": 0.0}
    no_env = ruler.ruler_chat_prompt(line, name)                                                      # without the run's variables nothing is marked
    assert not no_env.specific
    with _exact_env(10**6) as _:                                                                       # a name that is not a chat task name: untouched
        assert not ruler.ruler_chat_prompt(line, "ruler_4096:qa_1|0").specific


def test_ruler_chat_rows_are_counted_in_one_batch_before_the_per_row_calls():
    """lighteval_task._get_docs_from_split hands the rows of a split to `ruler_chat_prompt.prefetch`: one batch call, then the per-row calls only look the counts up."""
    import os
    from unittest import mock

    from lighteval.tasks import exact_prompt

    rows = [{"input": "w " * i + "Q?", "gen_prefix": "A:", "outputs": ["x"], "length": 100} for i in (3, 5, 7)]
    name = "ruler_4096_chat:qa_1|0"
    with _exact_env(100000) as _:
        ex = exact_prompt.get()
        calls = []
        orig = ex.tok

        class Spy:
            chat_template = "x"

            def apply_chat_template(self, *a, **k):
                return orig.apply_chat_template(*a, **k)

            def __call__(self, texts, add_special_tokens=False):
                calls.append(len(texts))
                return orig(texts, add_special_tokens=add_special_tokens)

        ex.tok = Spy()
        ex.pm.tokenizer = ex.tok
        ruler.ruler_chat_prompt.prefetch(rows, name)
        assert calls == [3]
        assert [ruler.ruler_chat_prompt(r, name).specific["n_tokens"] for r in rows] == [3 + 1 + 3, 5 + 1 + 3, 7 + 1 + 3] and calls == [3]
        with mock.patch.dict(os.environ, {"TITAN_EVAL_MAX_MODEL_LEN": ""}):
            ruler.ruler_chat_prompt.prefetch(rows, "ruler_4096:qa_1|0")                       # not a chat name / no window: nothing counted
        assert calls == [3]


# ---------------------------------------------------------------------------
# NVIDIA RULER main (db43f92) fallbacks, no length switch of ours (owner 11-16): CWE words from the wonderwords list while it holds enough, else from the Randle list
# (common_words_extraction.py:86-89); QA documents once, repeated when more are asked for than exist (qa.py:130-141)
# ---------------------------------------------------------------------------


def _ww(n):
    return [f"ww{i}" for i in range(n)]


def _randle(n):
    return [f"rw{i}" for i in range(n)]


def _words_of(context):
    import re

    return re.findall(r"\b(?:ww|rw)\d+\b", context)


def _main_prompt(cwe_input):
    """the words of the task itself (the in-context example before it always uses the wonderwords list)"""
    return cwe_input.split("Below is a numbered list")[-1]


def test_cwe_example_uses_wonderwords_up_to_and_including_the_list_size():
    import random
    from unittest import mock

    words = _ww(50)

    def boom():
        raise AssertionError("the Randle list must not be read while the wonderwords list holds enough words")

    with mock.patch.object(ruler, "_get_randle_words", boom):
        for num in (10, 49, 50):  # NVIDIA: `if num_words <= len(words)`
            random.seed(3)
            context, common = ruler._cwe_get_example(num, words, 3, 1, 10)
            random.seed(3)
            assert common == random.sample(words, num)[:10]
            assert set(_words_of(context)) <= set(words) and set(common) <= set(words)


def test_cwe_example_falls_back_to_the_randle_list_when_more_words_are_asked_for_than_exist():
    import random
    from unittest import mock

    words, randle = _ww(50), _randle(400)
    with mock.patch.object(ruler, "_get_randle_words", lambda: randle):
        random.seed(11)
        context, common = ruler._cwe_get_example(51, words, 3, 1, 10)  # NVIDIA: `else: random.sample(randle_words, num_words)`
        random.seed(11)
        expected = random.sample(randle, 51)
        assert common == expected[:10], "the same draw (random.sample on the global RNG, over the Randle list) as NVIDIA"
        seen = _words_of(context)
        assert set(seen) <= set(randle) and not any(w.startswith("ww") for w in seen), "no mixing of the two lists"
        assert len(seen) == 10 * 3 + 41 * 1 and len(set(seen)) == 51, "10 common words x 3, 41 other words x 1, 51 distinct words"
        random.seed(11)
        ruler._cwe_get_example(300, words, 3, 1, 10)  # far beyond the wonderwords list
        with mock.patch.object(ruler, "_get_randle_words", lambda: randle[:100]):
            try:
                ruler._cwe_get_example(101, words, 3, 1, 10)
            except ValueError:
                pass
            else:
                raise AssertionError("random.sample raises beyond the longest list, as in NVIDIA")


class _WordTokenizer:
    """one token per whitespace-separated piece"""

    def __call__(self, text):
        return type("Tokens", (), {"input_ids": text.split()})()


def test_cwe_sizing_runs_past_the_end_of_the_wonderwords_list_into_the_randle_list():
    from unittest import mock

    words, randle = _ww(80), _randle(2000)
    with mock.patch.object(ruler, "_get_cwe_words", lambda: words), mock.patch.object(ruler, "_get_randle_words", lambda: randle):
        # the in-context example costs ~480 tokens, the task 2 tokens x (10 x 30 + 3 x (N - 10)) for N words: 1,400 tokens hold ~60 words, 3,000 tokens ~330 (more than the 80 wonderwords)
        short = ruler._cwe_generate_samples(_WordTokenizer(), 1400, num_samples=3, incremental=10)
        long = ruler._cwe_generate_samples(_WordTokenizer(), 3000, num_samples=3, incremental=10)
    short_words = {w for s in short for w in _words_of(_main_prompt(s["input"]))}
    long_words = {w for s in long for w in _words_of(_main_prompt(s["input"]))}
    assert short_words and all(w.startswith("ww") for w in short_words), "within the wonderwords list: unchanged, no Randle word"
    assert long_words and all(w.startswith("rw") for w in long_words), "beyond it: the Randle words only (NVIDIA draws the whole list from one source)"
    assert all(s["length"] <= 3000 - ruler.PROMPT_SAFETY_MARGIN for s in long)
    assert min(s["length"] for s in long) > 0.9 * 3000, "the budget is filled although the wonderwords list holds 80 words (the old generator stopped there)"


def test_cwe_sizing_stops_at_the_longest_list_instead_of_raising():
    from unittest import mock

    words, randle = _ww(50), _randle(70)
    with mock.patch.object(ruler, "_get_cwe_words", lambda: words), mock.patch.object(ruler, "_get_randle_words", lambda: randle):
        out = ruler._cwe_generate_samples(_WordTokenizer(), 10**6, num_samples=2, incremental=10)
    main = _words_of(_main_prompt(out[0]["input"]))
    assert len(out) == 2 and len(main) == 10 * 30 + 60 * 3 and len(set(main)) == 70, "70 words = the longest list: 10 common x 30 + 60 others x 3"


def test_qa_documents_are_repeated_when_more_are_asked_for_than_exist():
    import random

    docs = ["alpha doc", "beta doc", "gamma doc"]
    qas = [{"query": "Q?", "outputs": ["a"], "context": [1], "more_context": [0, 2]}]

    def asked(num_docs):
        random.seed(0)
        text, answers = ruler._qa_generate_input_output(0, num_docs, qas=qas, docs=docs)
        assert answers == ["a"]
        body = text.split("The following are given documents.\n\n", 1)[1].split("\n\nAnswer the question", 1)[0]
        return [part.split("\n", 1)[1] for part in body.split("\n\n")]

    before = list(docs)
    assert sorted(asked(2)) == sorted(["beta doc", "alpha doc"]) or sorted(asked(2)) == sorted(["beta doc", "gamma doc"]), "fewer than exist: the gold document and a sample, as before"
    assert sorted(asked(3)) == sorted(docs), "NVIDIA: `num_docs < len(DOCS)` is false at equality, the list once (a copy: the shared list is not shuffled in place)"
    assert sorted(asked(7)) == sorted((docs * 3)[:7]), "(DOCS * ceil(7/3))[:7]: alpha, beta, gamma, alpha, beta, gamma, alpha"
    assert asked(7).count("alpha doc") == 3 and asked(7).count("gamma doc") == 2
    assert "beta doc" in asked(4) and len(asked(4)) == 4
    assert docs == before, "the document list of the reader is never reordered"


def test_qa_sizing_runs_past_the_number_of_documents():
    from unittest import mock

    docs = [f"document number {i} " + "x " * 20 for i in range(30)]
    qas = [{"query": "Q?", "outputs": ["a"], "context": [i], "more_context": [(i + 1) % 30]} for i in range(30)]
    out = ruler._qa_generate_samples(_WordTokenizer(), docs, qas, 2000, num_samples=3)
    assert len(out) == 3
    assert max(s["length"] for s in out) <= 2000 - ruler.PROMPT_SAFETY_MARGIN
    assert min(s["length"] for s in out) > 0.9 * 2000, "the budget is filled although the 30 documents are only ~700 tokens (it used to stop at 30 documents)"
    for s in out:
        assert s["input"].count("Document ") > 30, "documents are repeated"


def test_cwe_and_qa_1_prepared_folders_need_the_current_data_revision():
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for length in (4096, 262144):
            for subset in ("cwe", "qa_1", "vt"):
                (base / str(length) / subset).mkdir(parents=True)
                (base / str(length) / subset / "dataset_dict.json").write_text("{}")
        pairs = lambda: ruler._missing_pairs(base, [4096, 262144], ["cwe", "qa_1", "vt"])  # noqa: E731
        assert pairs() == [(4096, "cwe"), (4096, "qa_1"), (262144, "cwe"), (262144, "qa_1")], "no manifest: the two source-limited subtasks are not current, vt (no source limit) is"
        (base / "manifest.json").write_text(json.dumps({"data": {"4096": {"cwe": {"gen_rev": ruler.DATA_REV}, "qa_1": {"gen_rev": "older"}}, "262144": {"cwe": {}}}}))
        assert pairs() == [(4096, "qa_1"), (262144, "cwe"), (262144, "qa_1")]
        assert ruler._stale_pairs(base, pairs() + [(8192, "cwe")]) == pairs(), "a missing folder is not stale"
        (base / "manifest.json").write_text("{not json")
        assert (4096, "cwe") in pairs(), "an unreadable manifest never makes old data current"
        st = ruler.prepared_status(str(base / "tok"), [4096], ["vt"])
        assert st["missing"] == [(4096, "vt")]


def test_prepare_archives_a_stale_folder_and_stamps_the_new_one():
    import json
    import tempfile
    from pathlib import Path
    from unittest import mock

    sample = {"input": "x", "outputs": ["y"], "length": 5, "max_length": 4096, "gen_prefix": "A:"}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "key"
        old = base / "4096" / "cwe"
        old.mkdir(parents=True)
        (old / "dataset_dict.json").write_text("{}")
        (old / "OLD").write_text("old data")
        (base / "manifest.json").write_text(json.dumps({"data": {"4096": {"cwe": {"n": 1}}}}))
        with mock.patch.object(ruler, "_get_cache_dir", lambda t: base), mock.patch.object(ruler, "_load_tokenizer", lambda t: object()), mock.patch.object(
            ruler, "_generate_subset", lambda subset, length, tok: [dict(sample)]
        ):
            ruler.ensure_ruler_cache("tok", [4096], ["cwe", "vt"], prepared_only=False)
            assert ruler._missing_pairs(base, [4096], ["cwe", "vt"]) == []
            man = json.loads((base / "manifest.json").read_text())
            assert man["data"]["4096"]["cwe"]["gen_rev"] == ruler.DATA_REV and man["generator"]["data_rev"] == ruler.DATA_REV
            assert "gen_rev" not in man["data"]["4096"]["vt"]
            assert man["sources"]["randle_words"]["sha256"] == ruler.RANDLE_WORDS_SHA256
            archived = Path(man["data"]["4096"]["cwe"]["replaced_folder"])
            assert archived.parent == base / "_archive" and (archived / "OLD").read_text() == "old data", "the old folder is moved, not deleted"
            assert not (base / "4096" / "cwe" / "OLD").exists() and (base / "4096" / "cwe" / "dataset_dict.json").exists()
            # nothing is regenerated, nothing archived again
            before = sorted(p.name for p in (base / "_archive").iterdir())
            ruler.ensure_ruler_cache("tok", [4096], ["cwe", "vt"], prepared_only=False)
            assert sorted(p.name for p in (base / "_archive").iterdir()) == before
        with mock.patch.object(ruler, "_get_cache_dir", lambda t: base):
            old2 = base / "8192" / "qa_1"
            old2.mkdir(parents=True)
            (old2 / "dataset_dict.json").write_text("{}")
            try:
                ruler.ensure_ruler_cache("tok", [8192], ["qa_1"], prepared_only=True)
            except RuntimeError as e:
                assert "older generator revision" in str(e) and "titan-eval prepare ruler" in str(e)
            else:
                raise AssertionError("RULER_PREPARED_ONLY must refuse a folder of an older revision")


def test_randle_words_file_is_checked_against_the_pinned_sha256():
    import os
    import tempfile
    from pathlib import Path
    from unittest import mock

    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "english_words.json"
        f.write_text('{"0": "a", "1": "aa"}')
        ruler._get_randle_words.cache_clear()
        try:
            with mock.patch.dict(os.environ, {ruler.RANDLE_WORDS_ENV: str(f)}):
                try:
                    ruler._get_randle_words()
                except ValueError as e:
                    assert ruler.RANDLE_WORDS_SHA256 in str(e)
                else:
                    raise AssertionError("a file that is not the pinned one must be refused")
            with mock.patch.dict(os.environ, {ruler.RANDLE_WORDS_ENV: str(Path(tmp) / "missing.json")}):
                try:
                    ruler._get_randle_words()
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError("a missing local copy must be an error, not a download")
        finally:
            ruler._get_randle_words.cache_clear()
    assert len(ruler.RANDLE_WORDS_SHA256) == 64 and ruler.RANDLE_WORDS_URL.endswith(f"{ruler.RANDLE_WORDS_COMMIT}/scripts/data/synthetic/json/english_words.json")
