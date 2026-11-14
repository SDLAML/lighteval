"""Opt-in seeded subsets (lighteval/tasks/sampling.py, titan_eval flag `data.sample_per_group=N`): the full official set is the default of every task that supports the flag.

Plain functions: run with pytest, or `python tests/unit/tasks/test_sampling.py`. No GPU. The synthetic tests need no data; the last ones read the pinned datasets from the
offline cache and skip when a dataset is not there.
"""

import os
import unittest
from unittest import mock

from datasets import Dataset, DatasetDict

from lighteval.tasks.sampling import ENV_SAMPLE_PER_GROUP, SeededSubset, apply_dataset_filter, sample_per_group, subset_filter
from lighteval.tasks.tasks import chat_math2 as cm2
from lighteval.tasks.tasks import chat_multilingual as cmu
from lighteval.tasks.tasks import chat_polymath as cp
from lighteval.tasks.tasks import long_more as lm


def env(n):
    return mock.patch.dict(os.environ, {ENV_SAMPLE_PER_GROUP: str(n)} if n is not None else {ENV_SAMPLE_PER_GROUP: ""}, clear=False)


def test_env_switch():
    with env(None):
        assert sample_per_group() is None
        assert subset_filter("s", lambda r: r["id"], ["test"]) is None, "no flag = no filter = the full set"
    with env(20):
        assert sample_per_group() == 20
        assert isinstance(subset_filter("s", lambda r: r["id"], ["test"]), SeededSubset)
    for bad in ("0", "-3", "x", "1.5"):
        with env(bad):
            try:
                sample_per_group()
            except ValueError as e:
                assert "positive integer" in str(e)
            else:
                raise AssertionError(bad)


def _ds(n, split="test", dup=False):
    return Dataset.from_dict({"id": [f"i{k % (n // 2) if dup else k:04d}" for k in range(n)], "v": list(range(n))})


def test_seeded_subset_keeps_n_and_is_reproducible():
    flt = SeededSubset("seed-a", 10, lambda r: r["id"], ["test"])
    a = flt.apply(DatasetDict({"test": _ds(100), "other": _ds(30)}))
    b = SeededSubset("seed-a", 10, lambda r: r["id"], ["test"]).apply(DatasetDict({"test": _ds(100), "other": _ds(30)}))
    assert len(a["test"]) == 10 and a["test"]["id"] == b["test"]["id"], "same seed, same rows"
    assert len(a["other"]) == 30, "only the evaluated split is drawn"
    c = SeededSubset("seed-b", 10, lambda r: r["id"], ["test"]).apply(_ds(100))
    assert c["id"] != a["test"]["id"], "another seed, another draw"
    assert SeededSubset("seed-a", 10, lambda r: r["id"], ["test"]).pick([{"id": k} for k in sorted(_ds(100)["id"])]) == frozenset(a["test"]["id"])


def test_a_group_smaller_than_n_keeps_everything():
    assert len(SeededSubset("s", 50, lambda r: r["id"], ["test"]).apply(DatasetDict({"test": _ds(20)}))["test"]) == 20


def test_all_rows_of_a_drawn_key_are_kept():
    ds = _ds(40, dup=True)              # 20 keys, each twice
    out = SeededSubset("s", 5, lambda r: r["id"], ["test"]).apply(ds)
    assert len(out) == 10 and len(set(out["id"])) == 5


def test_apply_dataset_filter_takes_both_kinds():
    ds = DatasetDict({"test": _ds(10)})
    assert len(apply_dataset_filter(ds, lambda r: r["v"] < 3)["test"]) == 3
    assert len(apply_dataset_filter(ds, SeededSubset("s", 4, lambda r: r["id"], ["test"]))["test"]) == 4


def test_mmmlu_key_is_unique_per_row():
    a = {"Subject": "anatomy", "Unnamed: 0": 5}
    b = {"Subject": "virology", "Unnamed: 0": 5}
    assert cmu._mmmlu_key(a) != cmu._mmmlu_key(b), "`Unnamed: 0` alone is the index inside a subject (1,534 values for 14,042 rows)"


def test_gsm_limit_is_the_official_rule():
    rows = []
    for op in (2, 3):
        for pct, template, mode in lm.GSM_FILTER:
            rows += [{"op": op, "template": template, "mode": mode, "i": k} for k in range(int(200 * pct))]
    ds = Dataset.from_list(rows)
    out = lm.GsmLimit(20).apply(ds)
    assert len(out) == 2 * 20
    for op in (2, 3):
        mine = [r for r in out if r["op"] == op]
        assert len(mine) == 20
        assert sum(1 for r in mine if r["template"] == "crazy_zootopia" and r["mode"] == "normalforward") == 8
        assert [r["i"] for r in mine if r["template"] == "crazy_zootopia" and r["mode"] == "normalforward"] == list(range(8)), "the FIRST rows of each group (a prefix)"
    assert len(lm.GsmLimit(200).apply(ds)) == len(ds) and len(lm.GsmLimit(1000).apply(ds)) == len(ds)
    assert len(lm.GsmLimit(100).apply(ds)) == 2 * 100


def test_default_is_the_full_set_for_every_supporting_task():
    with env(None):
        tables = cm2._build_tasks(), cmu._build_tasks(), cp._build_tasks()
        for table in tables:
            for t in table:
                assert t.hf_filter is None, f"{t.name}: a filter without the flag cuts the set"
        gsm = [t for t in lm.TASKS_TABLE if t.name.startswith("gsm_infinite_")]
        assert len(gsm) == 20 and all(t.hf_filter is None for t in gsm)
    with env(20):
        assert all(isinstance(t.hf_filter, lm.GsmLimit) and t.hf_filter.limit == 20 for t in lm._build_gsm_tasks()), "the flag reaches every GSM-Infinite task"


def test_flag_reaches_every_group():
    with env(7):
        for t in cm2._build_tasks():
            if t.name.startswith(("omega_chat:", "mmlu_prox_chat:")):
                assert isinstance(t.hf_filter, SeededSubset) and t.hf_filter.n == 7 and t.hf_filter.splits == (list(t.evaluation_splits)[0],), t.name
            else:
                assert t.hf_filter is None, t.name                       # IMO-AnswerBench: no groups, no flag
        for t in cmu._build_tasks():
            assert isinstance(t.hf_filter, SeededSubset) and t.hf_filter.n == 7, t.name
        for t in cp._build_tasks():
            if t.name.startswith(("polymath_chat:", "polymath_en_chat:", "mgsm_chat:")):
                assert isinstance(t.hf_filter, SeededSubset) and t.hf_filter.n == 7, t.name
        assert isinstance(lm.gsm_limit_filter(), lm.GsmLimit) and lm.gsm_limit_filter().limit == 7


def test_seed_strings_are_the_first_builds():
    """The first build drew with random.Random(f"<SEED>:<group>").sample(sorted keys, N); the flag uses the same strings, so N = 20 (OMEGA) / 200 (MMLU-ProX, INCLUDE) gives the first build's items."""
    with env(20):
        om = {t.name: t.hf_filter.seed for t in cm2._build_tasks() if t.name.startswith("omega_chat:")}
        assert om["omega_chat:explorative:algebra_func_area:test_in"] == "chat_math2-v1:omega:explorative:algebra_func_area:test_in"
        px = {t.name: t.hf_filter.seed for t in cm2._build_tasks() if t.name.startswith("mmlu_prox_chat:")}
        assert px["mmlu_prox_chat:de"] == "chat_math2-v1:mmlu_prox:de"
        inc = {t.name: t.hf_filter.seed for t in cmu._build_tasks() if t.name.startswith("include_chat:")}
        assert inc["include_chat:sqi"] == "chat_multilingual-v1:include:Albanian"


# ---------------------------------------------------------------------------------------------- with the pinned data (skip when not cached)
def _load(repo, name, revision, split):
    from lighteval.tasks.lighteval_task import load_dataset_pinned

    try:
        return load_dataset_pinned(repo, name, revision=revision, split=split)
    except Exception as e:                       # offline cache without the dataset
        raise unittest.SkipTest(f"{repo} not cached: {type(e).__name__}")


def test_mmmlu_n_is_n_rows_with_the_pinned_data():
    with env(200):
        t = next(t for t in cmu._build_tasks() if t.name == "mmmlu_chat:de")
        ds = _load(t.hf_repo, t.hf_subset, t.hf_revision, "test")
        assert len(ds) == 14042
        assert len(t.hf_filter.apply(DatasetDict({"test": ds}))["test"]) == 200


def test_mmlu_prox_and_full_sizes_with_the_pinned_data():
    t = next(t for t in cm2._build_tasks() if t.name == "mmlu_prox_chat:sw")
    ds = _load(t.hf_repo, t.hf_subset, t.hf_revision, "test")
    assert len(ds) == cm2.MMLU_PROX_PER_LANG_ITEMS == 11759
    with env(200):
        t = next(t for t in cm2._build_tasks() if t.name == "mmlu_prox_chat:sw")
        assert len(t.hf_filter.apply(DatasetDict({"test": ds}))["test"]) == 200


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok  {name}")
            except unittest.SkipTest as e:
                print(f"skip {name}: {e}")
