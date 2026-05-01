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
        lengths=[512],
        subsets=["niah_single_1", "vt", "cwe", "fwe", "qa_2"],
    )
    task_map = {task.name: task.generation_size for task in tasks}

    assert task_map["ruler_512:niah_single_1"] == 50
    assert task_map["ruler_512:vt"] == 50
    assert task_map["ruler_512:cwe"] == 100
    assert task_map["ruler_512:fwe"] == 50
    assert task_map["ruler_512:qa_2"] == 50


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
