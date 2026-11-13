"""Dataset pins (owner 2026-11-13): a pinned `hf_revision` is honoured also OFFLINE, and a dataset update cannot change a score silently.

Plain functions without fixtures: run with pytest, or `python tests/unit/tasks/test_dataset_pins.py`. The tests that read the real HF cache
skip when the datasets are not there (they run in a subprocess with HF_HUB_OFFLINE=1: `datasets` reads the offline flag at import).

The fact behind it (datasets 5.2): offline, `load_dataset(repo, name, revision=<anything>)` returns the most recently modified processed copy
under $HF_HOME/datasets/<repo>/<name>/0.0.0/<hash>/, even for a revision that does not exist. `load_dataset_pinned` checks the copy.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lighteval.tasks import lighteval_task
from lighteval.tasks.lighteval_task import (
    PinnedRevisionNotCached,
    load_dataset_pinned,
    resolve_revision,
)

A = "a" * 40
B = "b" * 40
HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
CACHE_ONLY = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}


class _Patch:
    """Minimal monkeypatch: set module attributes, restore at exit."""

    def __init__(self, **attrs):
        self.attrs, self.old = attrs, {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.old[k] = getattr(lighteval_task, k)
            setattr(lighteval_task, k, v)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(lighteval_task, k, v)


def _loaded(*copies):
    """A stand-in for a loaded DatasetDict whose Arrow files live in the given `<hash>` folders."""
    part = SimpleNamespace(cache_files=[{"filename": f"/c/datasets/o___r/default/0.0.0/{c}/r-train.arrow"} for c in copies])
    return {"train": part}


# ---- resolve_revision -------------------------------------------------------------------------------------------------------------------

def test_a_commit_hash_resolves_to_itself():
    assert resolve_revision("o/r", A) == A


def test_a_ref_resolves_through_the_hub_cache_and_a_missing_ref_fails():
    with tempfile.TemporaryDirectory() as tmp:
        refs = Path(tmp) / "datasets--o--r" / "refs"
        (refs / "refs" / "convert").mkdir(parents=True)
        (refs / "main").write_text(A + "\n")
        (refs / "refs" / "convert" / "parquet").write_text(B)
        old = os.environ.get("HF_HUB_CACHE")
        os.environ["HF_HUB_CACHE"] = tmp
        try:
            assert resolve_revision("o/r", "main") == A
            assert resolve_revision("o/r", "refs/convert/parquet") == B
            try:
                resolve_revision("o/r", "refs/pr/9")
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("an unresolvable ref must fail, not fall back to another revision")
        finally:
            os.environ.pop("HF_HUB_CACHE") if old is None else os.environ.__setitem__("HF_HUB_CACHE", old)


# ---- load_dataset_pinned: the decision logic, with stand-ins ---------------------------------------------------------------------------

def test_unpinned_and_non_hub_loads_are_plain_load_dataset():
    calls = []

    def fake(**kw):
        calls.append(kw)
        return _loaded(B)                                   # a copy that is not checked: no revision / a local folder

    with _Patch(load_dataset=fake, hub_offline=lambda: True):
        assert load_dataset_pinned("o/r", "default") == _loaded(B)
        assert load_dataset_pinned("json", "default", revision=A) == _loaded(B)
    assert calls == [{"path": "o/r", "name": "default", "revision": None}, {"path": "json", "name": "default", "revision": A}]


def test_a_ref_that_the_hub_cache_cannot_resolve_is_an_error_offline_only():
    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.get("HF_HUB_CACHE")
        os.environ["HF_HUB_CACHE"] = tmp                    # an empty hub cache: "main" does not resolve
        try:
            with _Patch(load_dataset=lambda **kw: _loaded(A), hub_offline=lambda: False):
                assert load_dataset_pinned("o/r", "default", revision="main") == _loaded(A)          # online the hub resolved it
            with _Patch(load_dataset=lambda **kw: _loaded(A), hub_offline=lambda: True):
                try:
                    load_dataset_pinned("o/r", "default", revision="main")
                except FileNotFoundError:
                    pass
                else:
                    raise AssertionError("offline, a ref that does not resolve must fail")
        finally:
            os.environ.pop("HF_HUB_CACHE") if old is None else os.environ.__setitem__("HF_HUB_CACHE", old)


def test_the_pinned_copy_or_an_unlabelled_dataset_is_returned_as_loaded():
    for offline in (True, False):
        with _Patch(load_dataset=lambda **kw: _loaded(A), hub_offline=lambda: offline):
            assert load_dataset_pinned("o/r", "default", revision=A) == _loaded(A)
    double = {"train": SimpleNamespace(cache_files=[])}     # in-memory dataset / test double: nothing to check
    with _Patch(load_dataset=lambda **kw: double, hub_offline=lambda: True):
        assert load_dataset_pinned("o/r", "default", revision=A) is double


def test_another_copy_is_never_returned_when_the_pinned_one_is_missing():
    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.get("HF_DATASETS_CACHE")
        os.environ["HF_DATASETS_CACHE"] = tmp
        try:
            import datasets.config as dc
            old_cfg = dc.HF_DATASETS_CACHE
            dc.HF_DATASETS_CACHE = tmp
            for offline in (True, False):               # False: the hub was unreachable but the offline flag was not set; `datasets` still answered from its cache
                with _Patch(load_dataset=lambda **kw: _loaded(B), hub_offline=lambda: offline):
                    try:
                        load_dataset_pinned("o/r", "default", revision=A)
                    except PinnedRevisionNotCached as e:
                        assert A in str(e) and B in str(e) and "OfflineModeIsEnabled" in str(e)      # the message enters the offline fallback of download_dataset_worker
                        assert isinstance(e, ConnectionError)
                    else:
                        raise AssertionError("the newest copy was returned for a pinned revision")
        finally:
            dc.HF_DATASETS_CACHE = old_cfg
            os.environ.pop("HF_DATASETS_CACHE") if old is None else os.environ.__setitem__("HF_DATASETS_CACHE", old)


# ---- the real HF cache (offline, in a subprocess) --------------------------------------------------------------------------------------------

def _offline(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], env=CACHE_ONLY, capture_output=True, text=True, timeout=600)


def _need(repo: str):
    if not (HUB / ("datasets--" + repo.replace("/", "--")) / "snapshots").is_dir():
        raise unittest.SkipTest(f"{repo} is not in the HF cache")


def test_offline_a_bogus_revision_fails_where_plain_load_dataset_succeeds():
    _need("lighteval/boolq_helm")
    out = _offline(
        "import json\n"
        "from datasets import load_dataset\n"
        "from lighteval.tasks.lighteval_task import load_dataset_pinned, PinnedRevisionNotCached\n"
        "plain = load_dataset('lighteval/boolq_helm', 'default', revision='0' * 40)\n"      # datasets ignores the revision offline
        "try:\n    load_dataset_pinned('lighteval/boolq_helm', 'default', revision='0' * 40); err = None\n"
        "except PinnedRevisionNotCached as e:\n    err = str(e)[:60]\n"
        "print(json.dumps({'plain_rows': len(plain['validation']), 'pinned_error': err}))\n")
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["plain_rows"] == 3270 and got["pinned_error"] and "OfflineModeIsEnabled" in got["pinned_error"], (got, out.stderr[-500:])


def test_offline_the_pinned_copy_is_read_even_when_a_newer_copy_exists():
    """openai/gsm8k has two processed copies with identical content; `datasets` returns the newer one (740312ad), the pin asks for the older."""
    _need("openai/gsm8k")
    arrow = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "datasets" / "openai___gsm8k" / "main" / "0.0.0"
    if not (arrow / "cc7b047b6e5bb11b4f1af84efc572db110a51b3c").is_dir() or not (arrow / "740312add88f781978c0658806c59bc2815b9866").is_dir():
        raise unittest.SkipTest("both processed copies of openai/gsm8k are needed")
    out = _offline(
        "import json, os\n"
        "from lighteval.tasks.lighteval_task import load_dataset_pinned\n"
        "res = {}\n"
        "for rev in ('cc7b047b6e5bb11b4f1af84efc572db110a51b3c', '740312add88f781978c0658806c59bc2815b9866'):\n"
        "    ds = load_dataset_pinned('openai/gsm8k', 'main', revision=rev)\n"
        "    res[rev] = [sorted({os.path.basename(os.path.dirname(f['filename'])) for f in ds[s].cache_files}) for s in ds] + [len(ds['test'])]\n"
        "print(json.dumps(res))\n")
    got = json.loads(out.stdout.strip().splitlines()[-1])
    for rev, (train, test, rows) in got.items():
        assert train == [rev] and test == [rev] and rows == 1319, (rev, got, out.stderr[-500:])


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
