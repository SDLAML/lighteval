"""Tests of the base_science domain-task fixes (owner 2026-11-13): chembench / xfinbench / geobench drop rules, headqa test split.

Plain functions without fixtures: run with pytest, or `python tests/unit/tasks/test_domain_fixes.py`. The tests that read the local
parquet files (data/titaneval, Git LFS) or the pinned EleutherAI/headqa snapshot skip when they are not available.
"""

import collections
import unittest
from pathlib import Path

import lighteval
from lighteval.tasks.tasks import chembench, domain_drops, geobench, headqa, xfinbench

DATA = Path(lighteval.__file__).resolve().parents[2] / "data" / "titaneval"


def _rows(name):
    import pandas as pd

    path = DATA / f"{name}.parquet"
    if not path.exists() or path.stat().st_size < 1000:      # an LFS pointer is a few hundred bytes
        raise unittest.SkipTest(f"{path} not available")
    rows = pd.read_parquet(path).to_dict("records")
    return [{**r, "choices": [str(c) for c in r["choices"]]} for r in rows]         # plain lists, as in a datasets row


def _config(module, name):
    return next(c for c in module.TASKS_TABLE if c.name == name)


# ---- KeepRows ------------------------------------------------------------------------------------------------------------------------

def test_keep_rows_filters_and_counts():
    keep = domain_drops.KeepRows("t-v1", lambda row: "odd" if row["x"] % 2 else None)
    assert [keep({"x": i}) for i in range(4)] == [True, False, True, False]
    assert keep.counts([{"x": i} for i in range(5)]) == collections.Counter({"kept": 3, "odd": 2})


def test_filter_cache_key_depends_on_the_rule_string():
    from datasets.fingerprint import Hasher

    reason = chembench.drop_reason
    a, b = domain_drops.KeepRows("rule-1", reason), domain_drops.KeepRows("rule-2", reason)
    assert Hasher.hash(a) != Hasher.hash(b)                        # a changed rule string => a new filtered cache file
    assert Hasher.hash(a) == Hasher.hash(domain_drops.KeepRows("rule-1", reason))
    # ... and so does a changed part of the rule under an unchanged rule string (the regex edit that served a stale cache)
    import re

    c1 = domain_drops.KeepRows("rule-1", reason, parts=[re.compile("a", re.I)])
    c2 = domain_drops.KeepRows("rule-1", reason, parts=[re.compile("ab", re.I)])
    assert Hasher.hash(c1) != Hasher.hash(c2) and c1.digest != c2.digest
    assert Hasher.hash(c1) == Hasher.hash(domain_drops.KeepRows("rule-1", reason, parts=[re.compile("a", re.I)]))
    assert geobench._KEEP.digest != chembench._KEEP.digest != xfinbench._KEEP.digest


# ---- chembench -----------------------------------------------------------------------------------------------------------------------

def test_chembench_config_and_rule():
    for suffix in ("cf", "mcf", "mcf_em"):
        config = _config(chembench, f"chembench:{suffix}")
        assert config.version == 1 and config.hf_filter is chembench._KEEP
    assert chembench.drop_reason({"domain": "chemistry_chemical_preference"}) == "chemical_preference"
    assert chembench.drop_reason({"domain": "chemistry_organic_chemistry"}) is None


def test_chembench_counts_on_the_local_file():
    rows = _rows("chembench")
    counts = chembench._KEEP.counts(rows)
    assert len(rows) == 2542 and counts == collections.Counter({"kept": 1542, "chemical_preference": 1000}), counts
    assert all(len(r["choices"]) == 2 for r in rows if r["domain"] == chembench.DROPPED_DOMAIN)
    assert all(0 <= r["answer_index"] < len(r["choices"]) for r in rows)


# ---- xfinbench -----------------------------------------------------------------------------------------------------------------------

def test_xfinbench_config_and_rule():
    for suffix in ("cf", "mcf", "mcf_em"):
        config = _config(xfinbench, f"xfinbench:{suffix}")
        assert config.version == 1 and config.hf_filter is xfinbench._KEEP
    row = {"question": "q", "choices": ["a", "b", "c"], "answer_index": 1}
    assert xfinbench.drop_reason(row) is None
    assert xfinbench.drop_reason({**row, "choices": ["nan"], "answer_index": 3}) == "invalid"
    assert len(xfinbench.FIGURE_KEYS) == 120 and all(len(k) == 16 for k in xfinbench.FIGURE_KEYS)
    assert xfinbench.figure_key(row) == xfinbench.figure_key(dict(row))             # deterministic
    assert xfinbench.figure_key(row) != xfinbench.figure_key({**row, "answer_index": 2})      # the gold is part of the key


def test_xfinbench_counts_on_the_local_file():
    rows = _rows("xfinbench")
    counts = xfinbench._KEEP.counts(rows)
    assert len(rows) == 592 and counts == collections.Counter({"kept": 468, "figure": 120, "invalid": 4}), counts
    keys = [xfinbench.figure_key(r) for r in rows]
    assert len(set(keys)) == len(rows)                                         # a key names exactly one row
    assert {k for k in keys if k in xfinbench.FIGURE_KEYS} == set(xfinbench.FIGURE_KEYS)       # every key matches a row
    # the first proposal's text rule must not have caught a figure row that the field rule keeps: every "attached" row is a figure row
    assert all(xfinbench.drop_reason(r) for r in rows if "attached" in r["question"].lower())
    assert all(r["domain"] != "finance_TU" for r in rows if xfinbench.drop_reason(r) == "figure")


# ---- geobench ------------------------------------------------------------------------------------------------------------------------

def _geo(question, choices=("a", "b", "c", "d", "e")):
    return geobench.drop_reason({"question": question, "choices": list(choices), "answer_index": 0})


def test_geobench_rule_on_synthetic_rows():
    assert _geo("Using the table below asdf, what is the wind chill?") == "placeholder"
    assert _geo("According to the table below, which country is largest?") == "table_figure"
    assert _geo("Which conclusion can be drawn from Figure 2?") == "table_figure"
    assert _geo("Looking at the graph shown, which is true?") == "table_figure"
    assert _geo("The cities circled on the map all share which characteristic?") == "exhibit"
    assert _geo("Based on the population pyramid above, which stage is Angola in?") == "exhibit"
    assert _geo("The place represented in Source A is") == "exhibit"
    assert _geo("In the ring labeled B, land prices are") == "exhibit"
    assert _geo("This urban model best resembles a") == "exhibit"
    assert _geo("These religious structures do NOT share which characteristic?") == "exhibit"
    assert _geo("The J-curve, depicted in the image below, illustrates") == "exhibit"
    assert _geo("Which of the following is correct?", ("a", "", "c", "d", "e")) == "empty_option"
    assert _geo("Which of the following is correct?", ("a", "B", "b ", "d", "e")) == "duplicate_option"
    # generic wording stays
    assert _geo("Aquifers that form a water table separating unsaturated and saturated zones are called") is None
    assert _geo("Isolation, genetic drift, and founder effect help biologists figure out") is None
    assert _geo("Which of the following maps would likely have the largest scale?") is None
    assert _geo("The study of water on, in, and above the surface of the Earth is called") is None
    assert _geo("Which of the following examples would best represent the concept of map scale?") is None


def test_geobench_config():
    for suffix in ("cf", "mcf", "mcf_em"):
        config = _config(geobench, f"geobench:{suffix}")
        assert config.version == 1 and config.hf_filter is geobench._KEEP
    assert "K2" in geobench.__doc__ and "NOT GeoBench-VLM" in geobench.__doc__


def test_geobench_counts_on_the_local_file():
    rows = _rows("geobench")
    counts = geobench._KEEP.counts(rows)
    assert len(rows) == 1392 and counts == collections.Counter(
        {"kept": 1237, "exhibit": 103, "table_figure": 25, "placeholder": 21, "empty_option": 3, "duplicate_option": 3}
    ), counts
    assert all(geobench.drop_reason(r) for r in rows if "asdf" in r["question"].lower())
    kept = [r for r in rows if geobench.drop_reason(r) is None]
    assert all(r["choices"] and all(str(c).strip() for c in r["choices"]) for r in kept)
    assert all(len({str(c).strip().casefold() for c in r["choices"]}) == len(r["choices"]) for r in kept)


# ---- the loaded datasets (a stale filtered cache file would show here) -------------------------------------------------------------

def test_loaded_datasets_have_exactly_the_kept_rows():
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.registry import Registry

    for spec, module, name, kept in (
        ("chembench:mcf_em|5", chembench, "chembench", 1542),
        ("xfinbench:mcf_em|5", xfinbench, "xfinbench", 468),
        ("geobench:mcf_em|5", geobench, "geobench", 1237),
    ):
        rows = _rows(name)
        try:
            tasks = Registry(tasks=spec, load_multilingual=False).load_tasks()
            LightevalTask.load_datasets(tasks)
        except Exception as error:  # dataset not cached / offline
            raise unittest.SkipTest(f"dataset not available: {type(error).__name__}: {str(error)[:120]}")
        loaded = list(tasks[spec].dataset["test"]["question"])
        assert len(loaded) == kept, (spec, len(loaded))
        assert loaded == [r["question"] for r in rows if module.drop_reason(r) is None], spec
        assert len(tasks[spec].eval_docs()) == kept and len(tasks[spec].fewshot_docs()) == kept


# ---- headqa --------------------------------------------------------------------------------------------------------------------------

def test_headqa_config_is_the_pinned_english_test_split_with_validation_shots():
    for suffix in ("cf", "mcf", "mcf_em"):
        c = _config(headqa, f"headqa:{suffix}")
        assert (c.hf_repo, c.hf_subset, c.hf_revision) == ("EleutherAI/headqa", "en", "3860ebd7b109f45dfa97a9c8cb31b564165ca0f1")
        assert list(c.evaluation_splits) == ["test"]
        assert c.few_shots_split == "validation" and c.few_shots_select == "random_sampling"       # 4-option shots like the test items, never from the scored split
        assert c.version == 2 and c.hf_filter is None
    assert _config(headqa, "headqa:mcf_em").generation_size == 1


def test_headqa_gold_goes_through_the_answer_id_and_never_defaults():
    line = {"qtext": "Which?", "ra": 3, "answers": [{"aid": 1, "atext": "w"}, {"aid": 2, "atext": "x"}, {"aid": 3, "atext": "y"}, {"aid": 4, "atext": "z"}]}
    doc = headqa._headqa_mcf_prompt(line, "t")
    assert doc.gold_index == 2 and doc.choices == [" A", " B", " C", " D"]
    assert doc.query == "Question: Which?\n A. w\n B. x\n C. y\n D. z\nAnswer:"
    assert headqa._headqa_cf_prompt(line, "t").choices == [" w", " x", " y", " z"]
    try:
        headqa._gold({**line, "ra": 9})
        raise AssertionError("a row whose ra names no answer must raise")
    except ValueError:
        pass


def test_headqa_data_split_sizes_and_the_shots_are_validation_questions():
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.registry import Registry

    try:
        tasks = Registry(tasks="headqa:mcf_em|5", load_multilingual=False).load_tasks()
        LightevalTask.load_datasets(tasks)
    except Exception as error:  # dataset not cached / offline
        raise unittest.SkipTest(f"dataset not available: {type(error).__name__}: {str(error)[:120]}")
    task = tasks["headqa:mcf_em|5"]
    assert task.version == 2
    assert {k: len(v) for k, v in task.dataset.items()} == {"train": 2657, "validation": 1366, "test": 2742}
    docs, pool = task.eval_docs(), task.fewshot_docs()
    assert len(docs) == 2742 and len(pool) == 1366 and task.fewshot_split == "validation"
    assert {len(r["answers"]) for r in task.dataset["test"]} == {4} == {len(r["answers"]) for r in task.dataset["validation"]}    # train has 5
    key = lambda r: (r["qtext"].strip(), tuple(a["atext"].strip() for a in r["answers"]))      # noqa: E731
    assert not ({key(r) for r in task.dataset["test"]} & {key(r) for r in task.dataset["train"]})
    assert len({key(r) for r in task.dataset["test"]} & {key(r) for r in task.dataset["validation"]}) == 1      # the one shared question
    assert all(0 <= d.gold_index < len(d.choices) for d in docs)
    # no prompt carries its own item as a shot, and every shot is a validation question
    val_queries = {d.query for d in pool}
    for doc in task.get_docs(max_samples=200):
        assert len(doc.fewshot_samples) == 5
        assert all(shot.query in val_queries and shot.query.strip() != doc.query.strip() for shot in doc.fewshot_samples)


if __name__ == "__main__":
    ran = skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                ran += 1
                print("ok  ", name)
            except unittest.SkipTest as skip:
                skipped += 1
                print("skip", name, skip)
    print(f"{ran} passed, {skipped} skipped")
