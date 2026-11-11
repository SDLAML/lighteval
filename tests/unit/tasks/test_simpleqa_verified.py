import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.multilingual.tasks import simpleqa_verified
from lighteval.tasks.registry import Registry


CONFIGS = {config.name: config for config in simpleqa_verified.TASKS_TABLE}
LANGUAGES = ("ces", "dan", "deu", "fra", "ita", "nld", "pol", "spa")


@pytest.fixture
def record():
    return {
        "id": "simpleqa_verified_42",
        "question": " Welche Stadt? ",
        "answer": " München ",
        "answer_aliases": [" Munich ", "München", "", "   ", "Munich"],
        "easy_distractors": ["Paris", "London", "Rom", "Madrid"],
        "hard_distractors": ["Berlin", "Hamburg", "Bremen", "Köln"],
    }


@pytest.mark.parametrize("language", LANGUAGES)
def test_prompts_accept_aliases_and_score_only_canonical_answer_for_bpb(record, language):
    original = deepcopy(record)
    gen = CONFIGS[f"simpleqa_verified:{language}:gen"]
    bpb = CONFIGS[f"simpleqa_verified:{language}:bpb"]
    gen_doc = gen.prompt_function(record, gen.name)
    bpb_doc = bpb.prompt_function(record, bpb.name)

    assert gen_doc.query == bpb_doc.query
    assert "Welche Stadt?" in gen_doc.query
    assert "München" not in gen_doc.query
    assert gen_doc.get_golds() == [" München", " Munich"]
    assert bpb_doc.get_golds() == [" München"]
    assert gen_doc.choices == gen_doc.get_golds()
    assert bpb_doc.choices == bpb_doc.get_golds()
    assert gen_doc.task_name == gen.name
    assert bpb_doc.task_name == bpb.name
    assert record == original


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("variant", ["gen", "bpb"])
@pytest.mark.parametrize("answer", ["v0.4.0", "da Vinci Arts Middle School", "říjen 2023"])
def test_qa_references_preserve_answer_casing(record, language, variant, answer):
    record.update(answer=f" {answer} ", answer_aliases=[" an alternative answer "])
    original = deepcopy(record)
    config = CONFIGS[f"simpleqa_verified:{language}:{variant}"]
    doc = config.prompt_function(record, config.name)

    expected = [f" {answer}"]
    if variant == "gen":
        expected.append(" an alternative answer")
    assert doc.get_golds() == expected
    if variant == "bpb":
        prepared = config.metrics[0].sample_level_fn.prepare(doc, ModelResponse(logprobs=[-1.0]))
        assert prepared.weights == len(expected[0].encode("utf-8"))
    assert record == original


@pytest.mark.parametrize("aliases", [[], None])
def test_canonical_answer_is_valid_without_aliases(record, aliases):
    record["answer_aliases"] = aliases
    config = CONFIGS["simpleqa_verified:deu:gen"]
    assert config.prompt_function(record, config.name).get_golds() == [" München"]


@pytest.mark.parametrize("variant", ["gen", "bpb", "easy:mcf", "hard:mcf", "easy:mcf_em", "hard:mcf_em"])
def test_empty_canonical_answer_is_skipped(record, variant):
    record["answer"] = "   "
    config = CONFIGS[f"simpleqa_verified:deu:{variant}"]
    assert config.prompt_function(record, config.name) is None


@pytest.mark.parametrize(
    ("prediction", "em", "f1"),
    [
        ("MÜNCHEN!", 1.0, 1.0),
        ("munich", 1.0, 1.0),
        ("Berlin", 0.0, 0.0),
        ("München oder Berlin", 0.0, 0.5),
        ("", 0.0, 0.0),
    ],
)
def test_generative_metrics_score_aliases_and_reject_distractors(record, prediction, em, f1):
    config = CONFIGS["simpleqa_verified:deu:gen"]
    doc = config.prompt_function(record, config.name)
    scores = [metric.sample_level_fn.compute(doc, ModelResponse(text=[prediction])) for metric in config.metrics]
    assert scores == pytest.approx([em, f1])


@pytest.mark.parametrize(
    ("language", "answer", "alias"),
    [
        ("deu", "1 km", "ein Kilometer"),
        ("deu", "1", "einer"),
        ("fra", "1861", "l'année 1861"),
        ("fra", "1", "un"),
        ("nld", "1967", "het jaar 1967"),
        ("spa", "1965", "el año 1965"),
        ("ita", "I", "I"),
    ],
)
def test_generative_metrics_accept_short_references_and_aliases_in_any_case(record, language, answer, alias):
    record.update(answer=answer, answer_aliases=[alias])
    config = CONFIGS[f"simpleqa_verified:{language}:gen"]
    doc = config.prompt_function(record, config.name)

    for prediction in (alias, alias.lower(), alias.upper()):
        scores = [metric.sample_level_fn.compute(doc, ModelResponse(text=[prediction])) for metric in config.metrics]
        assert scores == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize(
    ("answer", "aliases", "prediction"),
    [
        ("29.1", ["29.1.0", "verze 29.1"], "291"),
        ("29.1", ["29.1.0", "verze 29.1"], "2910"),
        ("1.5", [], "15"),
        ("1,5", [], "15"),
        ("2:1", [], "21"),
        ("1/2", [], "12"),
        ("10-12", [], "1012"),
        ("-5", [], "5"),
        (".5", [], "5"),
        ("5%", [], "5"),
    ],
)
def test_generative_metrics_preserve_meaningful_numeric_punctuation(record, language, answer, aliases, prediction):
    record.update(answer=answer, answer_aliases=aliases)
    config = CONFIGS[f"simpleqa_verified:{language}:gen"]
    doc = config.prompt_function(record, config.name)

    correct_scores = [metric.sample_level_fn.compute(doc, ModelResponse(text=[answer])) for metric in config.metrics]
    assert correct_scores == pytest.approx([1.0, 1.0])
    em, f1 = [metric.sample_level_fn.compute(doc, ModelResponse(text=[prediction])) for metric in config.metrics]
    assert em == 0.0
    assert f1 < 1.0


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("prediction", ["29.1!", "29.1.", '"29.1"', " 29.1\n"])
def test_generative_metrics_ignore_surrounding_punctuation_for_numeric_answers(record, language, prediction):
    record.update(answer="29.1", answer_aliases=[])
    config = CONFIGS[f"simpleqa_verified:{language}:gen"]
    doc = config.prompt_function(record, config.name)
    scores = [metric.sample_level_fn.compute(doc, ModelResponse(text=[prediction])) for metric in config.metrics]
    assert scores == pytest.approx([1.0, 1.0])


def test_german_query_uses_localized_labels(record):
    config = CONFIGS["simpleqa_verified:deu:gen"]
    assert config.prompt_function(record, config.name).query == "Frage: Welche Stadt?\nAntwort:"


@pytest.mark.parametrize("variant", ["gen", "bpb", "easy:mcf", "hard:mcf", "easy:mcf_em", "hard:mcf_em"])
def test_eval_split_supports_fewshots_without_reusing_the_evaluation_row(record, variant):
    config = replace(CONFIGS[f"simpleqa_verified:deu:{variant}"], num_fewshots=1)
    task = LightevalTask(config=config)
    task.dataset = DatasetDict({"eval": Dataset.from_list([record, {**record, "question": "Welche andere Stadt?"}])})
    docs = task.get_docs()
    assert len(docs) == 2
    for doc in docs:
        assert len(doc.fewshot_samples) == 1
        assert doc.fewshot_samples[0].source_id != doc.source_id
        sample = doc.fewshot_samples[0]
        if variant.endswith(("mcf", "mcf_em")):
            assert f"\n {sample.get_golds()[0].strip()}. München\n" in sample.query
        else:
            assert sample.get_golds()[0] == " München"


def test_module_discovery_and_language_variant_groups(monkeypatch):
    configs = Registry._load_from_files([Path(simpleqa_verified.__file__)], "lighteval.tasks.multilingual.tasks")
    expected_names = {
        f"simpleqa_verified:{language}:{variant}"
        for language in LANGUAGES
        for variant in ("gen", "bpb", "easy:mcf", "hard:mcf", "easy:mcf_em", "hard:mcf_em")
    }
    assert set(configs) == expected_names
    monkeypatch.setattr(Registry, "load_all_task_configs", staticmethod(lambda **kwargs: configs))
    for selection, names in [
        ("simpleqa_verified", expected_names),
        ("simpleqa_verified:gen", {name for name in expected_names if name.endswith(":gen")}),
        ("simpleqa_verified:deu", {name for name in expected_names if name.startswith("simpleqa_verified:deu:")}),
        ("simpleqa_verified:mcf", {name for name in expected_names if name.endswith(":mcf")}),
        ("simpleqa_verified:mcf_em", {name for name in expected_names if name.endswith(":mcf_em")}),
        ("simpleqa_verified:deu:easy:mcf_em", {"simpleqa_verified:deu:easy:mcf_em"}),
        ("simpleqa_verified:deu:hard:mcf_em", {"simpleqa_verified:deu:hard:mcf_em"}),
        ("simpleqa_verified:deu:easy:mcf", {"simpleqa_verified:deu:easy:mcf"}),
        ("simpleqa_verified:deu:hard:mcf", {"simpleqa_verified:deu:hard:mcf"}),
        ("simpleqa_verified:deu:gen", {"simpleqa_verified:deu:gen"}),
    ]:
        assert set(Registry(f"{selection}|0", load_multilingual=True).load_tasks()) == {f"{name}|0" for name in names}


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_mcf_prompts_use_only_matching_distractors_and_map_gold_to_letter(record, language, difficulty):
    original = deepcopy(record)
    rng_state = random.getstate()
    config = CONFIGS[f"simpleqa_verified:{language}:{difficulty}:mcf"]
    doc = config.prompt_function(record, config.name)

    assert doc.task_name == config.name
    assert doc.choices == [f" {label}" for label in "ABCDE"]
    gold = doc.get_golds()[0].strip()
    assert f"\n {gold}. München\n" in doc.query
    for distractor in record[f"{difficulty}_distractors"]:
        assert f". {distractor}\n" in doc.query
    other = "hard" if difficulty == "easy" else "easy"
    assert all(distractor not in doc.query for distractor in record[f"{other}_distractors"])
    assert "Munich" not in doc.query
    assert random.getstate() == rng_state
    assert record == original


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_mcf_order_is_stable_for_eval_fewshots_and_rows_without_ids(record, difficulty):
    config = CONFIGS[f"simpleqa_verified:deu:{difficulty}:mcf"]
    for with_id in (True, False):
        row = dict(record)
        if not with_id:
            row.pop("id")
        doc = config.prompt_function(row, config.name)
        fewshot = config.prompt_function({**row, "__few_shots": True, "__index": 999}, config.name)
        assert fewshot.query == doc.query
        assert fewshot.gold_index == doc.gold_index


def test_mcf_correct_letter_varies_between_rows_and_matches_across_difficulties(record):
    letters = set()
    for index in range(50):
        row = {**record, "id": f"simpleqa_verified_{index}"}
        easy = CONFIGS["simpleqa_verified:deu:easy:mcf"]
        hard = CONFIGS["simpleqa_verified:deu:hard:mcf"]
        easy_doc = easy.prompt_function(row, easy.name)
        hard_doc = hard.prompt_function(row, hard.name)
        assert easy_doc.get_golds() == hard_doc.get_golds()
        letters.add(easy_doc.get_golds()[0].strip())
    assert letters == set("ABCDE")


@pytest.mark.parametrize("difficulty", ["easy", "hard"])
@pytest.mark.parametrize("winning_index", range(5))
def test_mcf_metrics_score_the_shuffled_gold_letter(record, difficulty, winning_index):
    config = CONFIGS[f"simpleqa_verified:deu:{difficulty}:mcf"]
    doc = config.prompt_function(record, config.name)
    logprobs = [-10.0] * 5
    logprobs[winning_index] = -0.1
    response = ModelResponse(logprobs=logprobs)
    scores = {metric.metric_name: metric.sample_level_fn.compute(doc, response) for metric in config.metrics}
    expected = float(winning_index in doc.gold_index)
    assert scores == {"acc": expected, "acc_norm": expected}


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("difficulty", ["easy", "hard"])
def test_mcf_em_generates_and_scores_the_same_shuffled_label_as_mcf(record, language, difficulty):
    mcf = CONFIGS[f"simpleqa_verified:{language}:{difficulty}:mcf"]
    config = CONFIGS[f"simpleqa_verified:{language}:{difficulty}:mcf_em"]
    mcf_doc = mcf.prompt_function(record, mcf.name)
    doc = config.prompt_function(record, config.name)

    assert doc.task_name == config.name
    assert doc.query == mcf_doc.query
    assert doc.choices == mcf_doc.choices
    assert doc.gold_index == mcf_doc.gold_index
    assert config.generation_size == 5
    assert config.stop_sequence == ["\n"]
    assert [metric.metric_name for metric in config.metrics] == ["em"]

    metric = config.metrics[0].sample_level_fn
    gold = doc.get_golds()[0].strip()
    for label in "ABCDE":
        assert metric.compute(doc, ModelResponse(text=[label])) == float(label == gold)
    assert metric.compute(doc, ModelResponse(text=[f" {gold}\n"])) == 1
    for prediction in ("München", gold.lower(), f"{gold}.", f"{gold} because it is correct", ""):
        assert metric.compute(doc, ModelResponse(text=[prediction])) == 0
