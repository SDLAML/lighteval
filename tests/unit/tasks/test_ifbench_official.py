"""IFBench scoring = the official allenai/IFBench code at a pinned commit (owner 11-16, chat review E).

`src/lighteval/tasks/tasks/ifbench/{instructions,instructions_util,instructions_registry,classic_instructions,evaluation_lib}.py` are the files of
github.com/allenai/IFBench main @ 488918665d53f867fa908a293769ec6e71e74aec (2026-11-13, Apache-2.0) with ONE change: the `from ifbench import ...` lines.
Plain functions: `HF_HUB_OFFLINE=1 python tests/unit/tasks/test_ifbench_official.py` (or pytest). Nothing here needs a GPU; the dataset test reads the pinned dataset from the
Hugging Face cache (offline) and skips when it is not there.
"""

import hashlib
import json
import os
import unittest
from pathlib import Path

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks import chat_evals
from lighteval.tasks.tasks.ifbench import classic_instructions, instructions, instructions_registry, instructions_util
from lighteval.tasks.tasks.ifbench.main import IFBench


PIN = "488918665d53f867fa908a293769ec6e71e74aec"
# sha256 of the OFFICIAL files at the pin (what `git show <pin>:<path> | sha256sum` gives)
OFFICIAL_SHA256 = {
    "instructions.py": "d3390118284ad4679f3c202ecfcd55811dc3e5a99fd94fced7320a7e718dcfdf",
    "instructions_util.py": "6996a6efaf6e3c92a2e926881eadc8c0262fb8d06c8cb56394ef7322f69e1930",
    "instructions_registry.py": "5119f2919261c77566f698f2f367883aff56cf89443bac5a56b6e30ddbd5aef9",
    "classic_instructions.py": "d26e642f2e71e4f57218e5e05f098a22e36311485779baac401161d3228c9973",
    "evaluation_lib.py": "3c37d11cf10718ec90ca14c5873a8ae7d1b666417a39ded498f835699961b533",
}
# the official data/IFBench_test.jsonl (the file README.md and run_eval.py name): sha256 of the file, and of its (key, instruction_id_list, kwargs) as canonical JSON / of its stripped prompts
OFFICIAL_DATA_FILE_SHA256 = "d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a"        # (recorded for the notes; the file itself is not in this repo)
OFFICIAL_DATA_RULES_SHA256 = "7e1ff7c2d307ca5ecb9dddc7fe7fb6cab0e7f2faf73cbd0a22a962f8d4fea5f0"
OFFICIAL_DATA_PROMPTS_STRIPPED_SHA256 = "7b10c59378a904725ddfa9f9cddf9771c80bd5c47764bab7c80361258ee51bc5"
DATASET_REVISION = "ed7c12452594b7028b15807b86d3938150b663ae"        # allenai/IFBench_test (= the official ifbench/data/IFBench_test.jsonl, bundled in the wheel)
PORT_DIR = Path(instructions.__file__).resolve().parent


def test_ported_files_are_the_official_bytes_at_the_pin():
    for name, official in OFFICIAL_SHA256.items():
        raw = (PORT_DIR / name).read_bytes()
        head = raw.split(b"\n", 3)
        assert head[0].startswith(b"# PORT of github.com/allenai/IFBench @ " + PIN.encode()) and official.encode() in head[0], name
        assert head[1].startswith(b"# The ONLY change") and head[2].startswith(b"# tests/unit/tasks/test_ifbench_official.py"), name
        body = head[3].replace(b"from lighteval.tasks.tasks.ifbench.instructions import", b"from ifbench.instructions import").replace(b"from lighteval.tasks.tasks.ifbench import", b"from ifbench import")
        assert hashlib.sha256(body).hexdigest() == official, f"{name} is not the official file of the pin (only the import lines may differ)"


def test_registry_is_the_official_one():
    d = instructions_registry.INSTRUCTION_DICT
    assert len(d) == 83                                            # 25 classic IFEval keys + 58 IFBench OOD keys (README.md)
    ood = {k: c for k, c in d.items() if c.__module__.endswith(".instructions")}
    assert len(ood) == 58 and len(d) - len(ood) == 25 and all(c.__module__.endswith(".classic_instructions") for k, c in d.items() if k not in ood)
    assert all(issubclass(c, instructions.Instruction) for c in d.values()) and issubclass(classic_instructions.KeywordChecker, instructions.Instruction)
    assert "custom:european_capitals_sort" in ood and "words:words_position" in ood and instructions_util.generate_keywords(2)


def test_no_ifbench_checker_takes_the_prompt_argument():
    """evaluation_lib re-builds a checker with the prompt only when its args have a key `prompt`. The repeat checkers read `prompt_to_repeat` from the kwargs and no OOD checker has a `prompt`
    argument, so the 2 prompts whose dataset text keeps trailing whitespace (keys 13, 19) score the same as the official test file."""
    import inspect
    for iid, cls in instructions_registry.INSTRUCTION_DICT.items():
        if cls.__module__.endswith(".instructions"):
            assert "prompt" not in inspect.signature(cls(iid).build_description).parameters, iid


def _item(iid, kwargs, text):
    from lighteval.tasks.requests import Doc
    doc = Doc(task_name="t", query="P", choices=[""], gold_index=0, specific={"instruction_id_list": [iid], "kwargs": [kwargs]})
    return IFBench().compute(doc, ModelResponse(text=[text], text_post_processed=[text]))


def test_the_metric_is_the_official_strict_then_loose_flow():
    # loose strips `*` and the first / last line (8 variants); strict does not
    out = _item("format:list", {"sep": "..."}, "**...** one\n**...** two\n**...** three")
    assert out["prompt_level_strict_acc"] == 0 and out["prompt_level_loose_acc"] == 1 and out["inst_level_strict_acc"] == [False] and out["inst_level_loose_acc"] == [True]
    # an empty answer follows nothing, a None-valued kwarg (the dataset pads every kwargs dict with all keys) is dropped before the checker is built
    out = _item("format:list", {"sep": "...", "N": None, "word": None}, "")
    assert out["prompt_level_strict_acc"] == 0 and out["prompt_level_loose_acc"] == 0
    out = _item("format:list", {"sep": "...", "N": None, "word": None}, "... a\n... b")
    assert out["prompt_level_strict_acc"] == 1 and out["prompt_level_loose_acc"] == 1


def test_checkers_changed_since_the_old_snapshot_follow_the_official_code():
    # IncludeKeyword: whole word (the old snapshot matched the substring in "category")
    assert _item("sentence:keyword", {"word": "cat", "N": 1}, "The category is large. Another sentence.")["prompt_level_strict_acc"] == 0
    assert _item("sentence:keyword", {"word": "cat", "N": 1}, "The cat is large. Another sentence.")["prompt_level_strict_acc"] == 1
    # SpecialBulletPoints: the marker must START a line (the old snapshot counted it anywhere)
    assert _item("format:list", {"sep": "..."}, "One... two... three... four")["prompt_level_strict_acc"] == 0
    assert _item("format:list", {"sep": "..."}, "Here is a list:\n... one\n... two\n... three")["prompt_level_strict_acc"] == 1


def test_odd_answers_do_not_crash_the_fixed_checkers():
    # the old snapshot raised ZeroDivisionError / IndexError on these answers (checked 11-16: 69 raises on 58 checkers x 19 odd texts, 1 with the official code at the pin: sentence:increment
    # on blank lines, which evaluation_lib never reaches because it skips blank answers); the official code returns False
    for iid, kwargs, text in (("ratio:stop_words", {"percentage": 27.0}, "!!!"), ("format:emoji", {}, "!!!"), ("words:alphabet", {}, "!!!"),
                              ("ratio:overlap", {"percentage": 72.0, "reference_text": "Use induction to prove the claims."}, "?")):
        out = _item(iid, kwargs, text)
        assert out["prompt_level_strict_acc"] == 0 and out["prompt_level_loose_acc"] == 0, (iid, out)


def _canon_rules(rows):
    return hashlib.sha256(json.dumps([[r["key"], r["instruction_id_list"], r["kwargs"]] for r in rows], sort_keys=True).encode()).hexdigest()


def test_the_dataset_the_task_reads_is_the_official_test_file():
    """allenai/IFBench_test @ ed7c1245 (via the pinned loader, offline) = data/IFBench_test.jsonl of the pin: keys, instruction ids and kwargs type for type (ints stay ints, None padding as in the file);
    the prompts of the 2 rows 13 and 19 only keep trailing whitespace that the official file stripped (no checker reads the prompt: test above)."""
    from lighteval.tasks.lighteval_task import load_dataset_pinned
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        ds = load_dataset_pinned("allenai/IFBench_test", None, revision=DATASET_REVISION)["train"]
    except Exception as e:      # noqa: BLE001 (not in the cache of this machine)
        raise unittest.SkipTest(f"allenai/IFBench_test @ {DATASET_REVISION[:8]} is not in the Hugging Face cache ({type(e).__name__})")
    rows = [dict(ds[i]) for i in range(len(ds))]
    assert len(rows) == 300 and _canon_rules(rows) == OFFICIAL_DATA_RULES_SHA256
    assert hashlib.sha256(json.dumps([r["prompt"].strip() for r in rows]).encode()).hexdigest() == OFFICIAL_DATA_PROMPTS_STRIPPED_SHA256
    assert [r["key"] for r in rows if r["prompt"] != r["prompt"].strip()] == ["13", "19"]
    # the registered task builds docs with exactly these kwargs
    doc = chat_evals.ifbench_chat_prompt(rows[284], "t")
    assert doc.specific["instruction_id_list"] == ["repeat:repeat_span"] and doc.specific["kwargs"][0]["n_start"] == 3 and isinstance(doc.specific["kwargs"][0]["n_start"], int)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok  {name}")
            except unittest.SkipTest as e:
                print(f"skip {name}: {e}")
