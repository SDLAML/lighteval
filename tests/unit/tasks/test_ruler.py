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
            assert chat.generation_size == base.generation_size == ruler._gen_size(sub, length), "RULER's own answer size per subtask"
            assert [m.metric_name for m in base.metrics] == ["ruler_match"] and [m.metric_name for m in chat.metrics] == [["ruler_match", "over_context"]], "the base scorer + the share not generated"
            assert base.prompt_function is ruler.ruler_prompt and chat.prompt_function is ruler.ruler_chat_prompt
            assert not chat.stop_sequence and not base.stop_sequence, "no stop sequence: RULER has none, a chat reply ends at the end of turn"
    assert tasks["ruler_4096_chat:niah_multivalue"].generation_size == 300 and tasks["ruler_8192_chat:niah_multivalue"].generation_size == 250
    assert tasks["ruler_8192_chat:cwe"].generation_size == 100 and tasks["ruler_8192_chat:vt"].generation_size == 50


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


def test_ruler_chat_item_that_does_not_fit_the_window_is_marked_and_scores_zero():
    """Owner 11-16 (`over_context: zero`, the OLMES rule): floor = the subtask's own answer size; the chat prompt size is estimated from the stored base length."""
    import os
    from unittest import mock

    from lighteval.models.model_output import ModelResponse

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return text.split(" ")

    # qa_1 at 4096: own answer size 50; stored length 4000 = base prompt 3950 + 50; the prefix "\nAnswer:" is ONE word for this tokenizer
    line = {"input": "Q?", "gen_prefix": "Answer:", "outputs": ["Paris"], "length": 4000}
    name = "ruler_4096_chat:qa_1|0"
    n_est = 4000 - 50 - 1 + 16                                                                       # answer_fit.CHAT_TEMPLATE_TOKENS
    fits_at = n_est + 50 + 64                                                                         # prompt + the 50-token answer + the 64-token margin
    with mock.patch.dict(os.environ, {"TITAN_EVAL_TOKENIZER": "fake", "TITAN_EVAL_MAX_MODEL_LEN": str(fits_at)}), mock.patch("lighteval.tasks.tasks.chat_long._tokenizer", return_value=Tok()):
        ok = ruler.ruler_chat_prompt(line, name)
    assert ok.specific["over_context"] is False and ok.specific["n_tokens"] == n_est and ok.specific["answer_budget"] == 50 and "no_generation" not in ok.specific and ok.query == "Q?"
    with mock.patch.dict(os.environ, {"TITAN_EVAL_TOKENIZER": "fake", "TITAN_EVAL_MAX_MODEL_LEN": str(fits_at - 1)}), mock.patch("lighteval.tasks.tasks.chat_long._tokenizer", return_value=Tok()):
        over = ruler.ruler_chat_prompt(line, name)
    assert over.specific["over_context"] and over.specific["no_generation"] and over.query == "Reply with the word OK." and over.specific["answer_budget"] == 0
    metric = ruler.ruler_chat_metric.sample_level_fn
    assert metric.compute(over, ModelResponse(text=["Paris"])) == {"ruler_match": 0.0, "over_context": 1.0}, "an item not generated scores 0 whatever came back"
    assert metric.compute(ok, ModelResponse(text=["It is Paris."])) == {"ruler_match": 1.0, "over_context": 0.0}
    no_env = ruler.ruler_chat_prompt(line, name)                                                      # without the run's variables nothing is marked
    assert not no_env.specific
