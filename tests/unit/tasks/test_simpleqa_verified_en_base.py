"""Tests of simpleqa_verified_base:en:gen, the English twin of simpleqa_verified_base:<lang>:gen (simpleqa_verified_en_base.py).

Plain functions without fixtures: run with pytest, or `python tests/unit/tasks/test_simpleqa_verified_en_base.py`. The tests that read the pinned
datasets skip when they are not in the Hugging Face cache.
"""

import hashlib
import unittest

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_judged, heldout_shots, simpleqa_verified_base as twin, simpleqa_verified_en_base as en
from lighteval.utils.language import Language


NAME = "simpleqa_verified_base:en:gen"
LANGUAGES = [language.value for language in twin._LANGUAGES]

# sha1 of "\n<<SEP>>\n".join(prompts) of simpleqa_verified_base:<lang>:gen|5 (get_docs order, plain text, no base-prompt setting), recorded on 2026-11-19 BEFORE the English
# task was added: the English task must not change one byte of what the twin sends. Change these only when the twin is changed on purpose (and its version is bumped).
TWIN_PROMPTS_SHA1 = {
    "ces": "00aef6184f9db01352fa61d11e037b941e5c58ed",
    "dan": "e341c86793c0fad1233204034de3ec13f0506971",
    "deu": "d6ba8bf78cfcaf05c1cef7e9a7106ce7daae8262",
    "fra": "bcd0ab1c093eb5f6efad0ad7631d4ad44b3e71c4",
    "ita": "d6217d3437d47f013fb4fcb52dedfbc3cfdfd9e0",
    "nld": "e1100fb4078a579c90224a0479f3fdcf72b7a627",
    "pol": "ea516d82e12bae491f9b3e08dcb1efae500fa51f",
    "spa": "476524784d6f4c551ace7e15c573b0b311c11ed9",
}
TWIN_SCORED = {"ces": 95, "dan": 95, "deu": 995, "fra": 295, "ita": 95, "nld": 95, "pol": 95, "spa": 95}


def _need(fn, *args):
    try:
        return fn(*args)
    except Exception as error:  # dataset not cached / offline
        raise unittest.SkipTest(f"dataset not available: {type(error).__name__}: {str(error)[:120]}")


def _english_rows():
    return _need(heldout_shots.load_rows, en.REPO, en.SUBSET, "eval", en.REVISION)


def _row(**kw):
    return {"original_index": 7, "problem": "Who wrote it?", "answer": "Ada", **kw}


# ---- the data and the held-out shots ------------------------------------------------------------------------------------------

def test_same_dataset_and_pin_as_the_chat_task():
    assert (en.REPO, en.SUBSET, en.REVISION) == (chat_judged.SQV_REPO, "simpleqa_verified", chat_judged.SQV_REVISION)
    assert len(en.REVISION) == 40


def test_held_out_shots_are_the_twins_and_are_found_exactly_once_in_the_english_split():
    assert en.SHOT_IDS is twin.SHOT_IDS and len(twin.SHOT_IDS) == 5
    rows = _english_rows()
    assert len(rows) == 1000 and len({r["original_index"] for r in rows}) == 1000
    shots = en.shot_rows()
    assert [en.row_id(r) for r in shots] == list(twin.SHOT_IDS)
    assert [r["answer_type"] for r in shots] == ["Person", "Date", "Place", "Number", "Other"]


def test_english_ids_are_the_translated_datasets_ids():
    """ellamind's id = "simpleqa_verified_" + original_index: for every row of the 8 languages the English row with that original_index exists and has the same metadata,
    so the five shot ids of the twin name the same five questions in English."""
    rows = {r["original_index"]: r for r in _english_rows()}
    for language in LANGUAGES:
        translated = _need(heldout_shots.load_rows, twin.REPO, language, "eval", twin.REVISION)
        for r in translated:
            assert r["id"] == f"simpleqa_verified_{r['original_index']}", (language, r["id"])
            e = rows[r["original_index"]]
            for key in ("topic", "answer_type", "multi_step", "requires_reasoning"):
                assert r[key] == e[key], (language, r["id"], key)


def test_held_out_rows_are_shown_never_scored():
    task = Registry(tasks=f"{NAME}|5", load_multilingual=False).load_tasks()[f"{NAME}|5"]
    eval_docs = _need(task.eval_docs)
    assert len(eval_docs) == 995
    shots = task.fewshot_docs()
    assert len(shots) == 5
    assert not {s.query for s in shots} & {d.query for d in eval_docs}
    docs = task.get_docs()
    assert all(len(d.fewshot_samples) == 5 for d in docs)
    assert [d.query for d in docs[0].fewshot_samples] == [s.query for s in shots]  # the same five, in the twin's order, for every question
    assert all([x.query for x in d.fewshot_samples] == [s.query for s in shots] for d in docs[:50])


def test_prompt_function_holds_out_a_shot_row_unless_it_builds_a_demonstration():
    config = {c.name: c for c in en.TASKS_TABLE}[NAME]
    held = _row(original_index=int(twin.SHOT_IDS[0].rsplit("_", 1)[1]))
    assert config.prompt_function(held, config.name) is None
    assert config.prompt_function({**held, "__few_shots": True}, config.name) is not None
    assert config.prompt_function(_row(), config.name) is not None


# ---- prompt, gold, range note ------------------------------------------------------------------------------------------------

def test_prompt_is_the_twins_cloze_template_with_the_english_literals():
    config = {c.name: c for c in en.TASKS_TABLE}[NAME]
    doc = config.prompt_function(_row(problem="who wrote it? ", answer=" Ada "), config.name)
    assert doc.query == "Question: Who wrote it?\nAnswer:"          # capitalised, stripped, no instruction
    assert doc.choices == [" Ada"] and doc.get_golds() == [" Ada"] and doc.gold_index == [0]
    # the same function as the twin's languages (only the language differs)
    deu = {c.name: c for c in twin.TASKS_TABLE}["simpleqa_verified_base:deu:gen"]
    german = deu.prompt_function({"id": "simpleqa_verified_1", "question": "Wer?", "answer": "Ada", "answer_aliases": []}, deu.name)
    assert german.query == "Frage: Wer?\nAntwort:"
    english_alias = en._get_qa_prompt_function(Language.ENGLISH, include_aliases=True)(
        {"question": "Who?", "answer": "Ada Lovelace", "answer_aliases": ["Ada", "Lovelace"]}, "t")
    assert english_alias.choices == [" Ada Lovelace", " Ada", " Lovelace"]       # the scorer's alias machinery (this split has no alias column)


def test_acceptance_range_note_is_removed_from_the_gold_only():
    cases = {
        "390 metres (acceptable range: anything between 386 and 394 metres)": "390 metres",
        "150 (acceptable range: anything between 148 and 152)": "150",
        "10,990,502 (Acceptable Range: anything between 10,440,976 and 11,540,028)": "10,990,502",
        "IIT (BHU)": "IIT (BHU)",
        "tramp steamer (or collier)": "tramp steamer (or collier)",
        "Version 3.2.6827 (aka 3.2.6760)": "Version 3.2.6827 (aka 3.2.6760)",
        "no note": "no note",
    }
    for given, expected in cases.items():
        assert en.strip_range_note(given) == expected, given
    config = {c.name: c for c in en.TASKS_TABLE}[NAME]
    doc = config.prompt_function(_row(answer="150 (acceptable range: anything between 148 and 152)"), config.name)
    assert doc.get_golds() == [" 150"]


def test_no_gold_of_the_pinned_split_keeps_a_range_note_or_is_empty():
    rows = _english_rows()
    noted = [r for r in rows if "acceptable range" in r["answer"].lower()]
    assert len(noted) == 88
    task = Registry(tasks=f"{NAME}|5", load_multilingual=False).load_tasks()[f"{NAME}|5"]
    docs = _need(task.eval_docs)
    golds = [d.get_golds()[0].strip() for d in docs]
    assert all(golds) and not any("acceptable range" in g.lower() or "anything between" in g.lower() for g in golds)
    assert sum(1 for r in noted if en.row_id(r) not in twin.SHOT_IDS) == 87            # 87 scored + demonstration 1202
    assert [s.get_golds()[0].strip() for s in task.fewshot_docs()] == ["Curchod", "1861", "Papua New Guinea and Indonesia", "53rd minute", "Liverpool 2 - 1 Leicester"]


# ---- metric --------------------------------------------------------------------------------------------------------------------

def test_metrics_are_the_twins_em_and_f1_and_score_the_gold_as_one():
    config = {c.name: c for c in en.TASKS_TABLE}[NAME]
    assert [m.metric_name for m in config.metrics] == ["em", "f1"]
    assert config.generation_size == 128 and config.stop_sequence == ["\n"] and config.few_shots_select == "sequential"
    em, f1 = (m.sample_level_fn for m in config.metrics)
    doc = config.prompt_function(_row(answer="33.7738 (acceptable range: anything between 33.7586 and 33.8022)"), config.name)
    for text, want_em in {" 33.7738": 1, "33.7738": 1, " 33.7738.": 1, " 33.7739": 0, " 33,7738": 0, "": 0}.items():
        assert em.compute(doc, ModelResponse(text=[text])) == want_em, text
    named = config.prompt_function(_row(answer="Papua New Guinea and Indonesia"), config.name)
    assert em.compute(named, ModelResponse(text=[" papua new guinea and indonesia"])) == 1
    assert em.compute(named, ModelResponse(text=[" Indonesia and Papua New Guinea"])) == 0           # no alias: order matters, as in the twin without an alias
    assert 0 < f1.compute(named, ModelResponse(text=[" Papua New Guinea"])) < 1
    assert f1.compute(named, ModelResponse(text=[" Fiji"])) == 0
    # alias-aware like the twin (max over the canonical answer and its aliases), here through the same English metric objects
    aliased = en._get_qa_prompt_function(Language.ENGLISH, include_aliases=True)({"question": "Who?", "answer": "Ada Lovelace", "answer_aliases": ["Lovelace"]}, "t")
    assert em.compute(aliased, ModelResponse(text=[" lovelace"])) == 1


def test_every_scored_gold_is_a_perfect_prediction_and_an_empty_answer_scores_zero():
    config = {c.name: c for c in en.TASKS_TABLE}[NAME]
    em, f1 = (m.sample_level_fn for m in config.metrics)
    task = Registry(tasks=f"{NAME}|5", load_multilingual=False).load_tasks()[f"{NAME}|5"]
    docs = _need(task.eval_docs)
    assert all(em.compute(d, ModelResponse(text=[d.get_golds()[0]])) == 1 for d in docs)
    assert all(f1.compute(d, ModelResponse(text=[d.get_golds()[0]])) == 1 for d in docs)
    assert not any(em.compute(d, ModelResponse(text=[""])) for d in docs)
    assert not any(f1.compute(d, ModelResponse(text=[""])) for d in docs)


# ---- the twin is untouched -------------------------------------------------------------------------------------------------------

def test_registration_leaves_the_twins_specs_at_the_eight_languages():
    twin_names = {f"simpleqa_verified_base:{lang}:gen|5" for lang in LANGUAGES}
    for spec in ("simpleqa_verified_base:gen|5", "simpleqa_verified_base|5"):
        assert set(Registry(tasks=spec, load_multilingual=False).load_tasks()) == twin_names, spec
    assert set(Registry(tasks=f"{NAME}|5", load_multilingual=False).load_tasks()) == {f"{NAME}|5"}
    assert [c.name for c in twin.TASKS_TABLE] == [f"simpleqa_verified_base:{lang}:gen" for lang in LANGUAGES]
    config = {c.name: c for c in en.TASKS_TABLE}[NAME]
    assert config.superset_keys == ()
    # a spec of the English task together with the twin's: nine tasks, each exactly once
    both = Registry(tasks=f"simpleqa_verified_base:gen|5,{NAME}|5", load_multilingual=False).load_tasks()
    assert len(both) == 9 and f"{NAME}|5" in both


def test_twin_prompts_are_byte_identical_to_the_recorded_ones():
    spec = "simpleqa_verified_base:gen|5"
    tasks = Registry(tasks=spec, load_multilingual=False).load_tasks()
    _need(LightevalTask.load_datasets, tasks)
    pm = PromptManager(use_chat_template=False)
    got = {}
    for name, task in tasks.items():
        prompts = [pm.prepare_prompt(d) for d in task.get_docs()]
        language = name.split(":")[1]
        got[language] = (len(prompts), hashlib.sha1("\n<<SEP>>\n".join(prompts).encode("utf-8")).hexdigest())
    assert {k: v[0] for k, v in got.items()} == TWIN_SCORED
    assert {k: v[1] for k, v in got.items()} == TWIN_PROMPTS_SHA1


def test_plain_text_prompt_of_the_english_task():
    task = Registry(tasks=f"{NAME}|5", load_multilingual=False).load_tasks()[f"{NAME}|5"]
    docs = _need(task.get_docs)
    prompt = PromptManager(use_chat_template=False).prepare_prompt(docs[0])
    parts = prompt.split("\n\n")
    assert len(parts) == 6 and all(p.startswith("Question: ") and "\nAnswer:" in p for p in parts)
    assert [p.split("\nAnswer: ")[1] for p in parts[:5]] == ["Curchod", "1861", "Papua New Guinea and Indonesia", "53rd minute", "Liverpool 2 - 1 Leicester"]
    assert parts[5].endswith("\nAnswer:") and "Marlow Medal" not in parts[5]
    assert isinstance(docs[0], Doc) and docs[0].stop_sequences == ["\n"] and docs[0].generation_size == 128


if __name__ == "__main__":
    import sys

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except unittest.SkipTest as skip:
                print("skip", name, skip)
            except Exception as error:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(error).__name__, error)
    sys.exit(1 if failed else 0)
