"""BeyondBench easy suite, base task `beyondbench_easy_base:<task>` (owner 11-18): the vendored official files, the protocol of the owner's scripts, the generated items, the scorer, the metric.

`HF_HUB_OFFLINE=1 python -m pytest tests/unit/tasks/test_beyondbench.py` (CPU only; about 1 minute). Tests that read the stored results of the owner's two official runs
(/e/data1/trustllmd/filatov1/eval_results/beyond_bench) skip when the folder is absent.
"""

import glob
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import pytest

from lighteval.tasks.tasks import beyondbench, beyondbench_items as bb, beyondbench_official as official


PKG = Path(official.__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
STORED = Path("/e/data1/trustllmd/filatov1/eval_results/beyond_bench")
GOLDEN = {                                  # (task, list size, index) -> (prompt, gold JSON): the first item of ten configurations, as in the stored prompts of both official runs
    ("sum", 8, 0): ("Add the following list of numbers:\n[63, -72, -94, 89, -30, -38, -43, -65]\n\nProvide the sum. Your final answer must be in the format \\boxed{answer} at the end.", "-190"),
    ("median", 9, 0): ("Find the median value of the following list of numbers:\n[63, -72, -94, 89, -30, -38, -43, -65, 88]\n\nThe median is the middle value when the list is sorted. If there is an even number of elements, the median is the average of the two middle values. Your final answer must be in the format \\boxed{median value} at the end.", "-38"),
    ("multiplication", 4, 0): ("Multiply the following list of numbers:\n[63, -72, -94, 89]\n\nProvide the product. Your final answer must be in the format \\boxed{answer} at the end.", "37948176"),
    ("absolute_difference", 2, 0): ("Find the absolute difference between the following list of numbers: \n[63, -72]\n\nProvide the result. Your final answer must be in the format \\boxed{answer} at the end.", "135"),
    ("division", None, 0): ("Divide 63 by -72.\n\nProvide the answer as a floating point number. Your final answer must be in the format \\boxed{answer} at the end.", "-0.875"),
    ("comparison", None, 0): ("Compare the following two numbers and determine their relationship:\n\nNumber 1: -98\nNumber 2: -98\n\nIs Number 1 greater than, less than, or equal to Number 2? Your final answer must be in the format \\boxed{relation} at the end, where 'relation' is one of: 'greater than', 'less than', or 'equal to'.", '"equal to"'),
    ("rotate_list", 8, 0): ("Rotate the list [63, -72, -94, 89, -30, -38, -43, -65] to the right by 6 position(s). In a right rotation by k, the last k elements move to the front.\n\nYour final answer must be in the format \\boxed{[v1, v2, ...]} at the end of your response.", "[-94,89,-30,-38,-43,-65,63,-72]"),
    ("element_frequency", 8, 0): ("Count the frequency of each unique element in the list [10, -7, -10, -2, -3, -3, -6, -7]. Return the result as a list of [value, count] pairs sorted by value in ascending order. For example, if 3 appears 2 times and 7 appears 1 time, the answer would be [[3, 2], [7, 1]].\n\nYour final answer must be in the format \\boxed{[[v1, c1], [v2, c2], ...]} at the end of your response.", "[[-10,1],[-7,2],[-6,1],[-3,2],[-2,1],[10,1]]"),
}


@pytest.fixture(scope="module")
def rows():
    return bb.build_items()


# ---------------------------------------------------------------------------------------------------------------- the vendored official files
def test_vendored_files_are_the_official_bytes():
    assert official.OFFICIAL_VERSION == "0.2.1" and len(official.OFFICIAL_FILES) == 58
    for rel, digest in official.OFFICIAL_FILES.items():
        assert hashlib.sha256((PKG / rel).read_bytes()).hexdigest() == digest, rel               # the sha256 of the wheel's RECORD
    assert hashlib.sha256((PKG / "LICENSE").read_bytes()).hexdigest() == official.OFFICIAL_LICENSE_SHA256
    standins = {"__init__.py", "core/__init__.py", "core/base_task.py", "utils/__init__.py", "parsers/__init__.py", "tasks/__init__.py", "tasks/easy/__init__.py"}
    found = {str(p.relative_to(PKG)) for p in PKG.rglob("*.py") if "__pycache__" not in p.parts}
    assert found == set(official.OFFICIAL_FILES) | standins                                      # nothing else is in the folder
    for s in sorted(standins - {"__init__.py"}):
        assert "stand-in" in (PKG / s).read_text().split("\n", 3)[0].lower() or s == "tasks/easy/__init__.py", s
    easy = {p.name[:-8] for p in (PKG / "tasks" / "easy").glob("*_task.py")}
    assert easy == set(bb.TASKS) and len(bb.TASKS) == 44 == len(set(bb.TASKS))


def test_official_wheel_matches_when_available():
    """If the wheel of beyondbench 0.2.1 is unpacked next to the review (tmp/bb_dl/src021) the vendored files equal the files in it (the RECORD hashes above already say so)."""
    src = Path("/e/scratch/trustllm-eu/wang55/tmp/bb_dl/src021/beyondbench")
    if not src.is_dir():
        pytest.skip("wheel not unpacked")
    for rel in official.OFFICIAL_FILES:
        assert (src / rel).read_bytes() == (PKG / rel).read_bytes(), rel


# ---------------------------------------------------------------------------------------------------------------- the protocol of the owner's scripts
def test_protocol_equals_the_owner_script():
    sh = (ROOT / "scripts" / "run_beyondbench.sh").read_text() if (ROOT / "scripts" / "run_beyondbench.sh").exists() else None
    if sh is None:
        pytest.skip("scripts/run_beyondbench.sh is not in this tree")
    for const, expect in (("DATAPOINTS", bb.DATAPOINTS), ("SEED", bb.SEED)):
        assert re.search(rf"^{const}={expect}$", sh, re.M), const
    assert 'SUITE="easy"' in sh and "--backend vllm" in sh and "--no-chat-template" in sh and "--temperature 0 " in sh and "--top-p 1 " in sh and f"--max-tokens {bb.MAX_NEW_TOKENS} " in sh
    sizes = {name: tuple(int(x) for x in v.split(",")) for name, v in re.findall(r'^(\w+)_SIZES="\$\{\w+:-([\d,]+)\}"', sh, re.M)}
    assert sizes == {"GENERAL": (8, 16), "MEDIAN": (9, 16), "ELEMENT_FREQUENCY": (8, 16), "MULTIPLICATION": (2, 4), "PAIRED_LIST": (5, 10), "SET_OPERATIONS": (6, 12)}
    general = re.search(r"general_tasks=\(\n(.*?)\n\)", sh, re.S).group(1).split()
    assert tuple(general) == bb.GENERAL_TASKS and len(general) == 33
    got = {g: (s, r, t) for g, s, r, t in bb.GROUPS}
    assert got["general"][:2] == ((8, 16), (-100, 100)) and got["median"][:2] == ((9, 16), (-100, 100)) and got["multiplication"][:2] == ((2, 4), (-100, 100)) and got["paired_lists"][:2] == ((5, 10), (-100, 100))
    assert got["element_frequency"][:2] == ((8, 16), (-10, 10)) and got["set_operations"][:2] == ((6, 12), (-20, 20)) and got["fixed_inputs"][:2] == ((2,), (-100, 100))
    assert "range_args=(--range-min -20 --range-max 20)" in sh.replace('"', "") or "--range-min -20 --range-max 20" in sh
    assert "--range-min -10 --range-max 10" in sh
    calls = {m.group(1): re.findall(r"[a-z_]+", m.group(2)) for m in re.finditer(r'run_group "(\w+)" "[^"]*"((?:\s*\\\n\s*[a-z_]+)+)', sh)}
    assert calls == {g: list(tasks) for g, _, _, tasks in bb.GROUPS if g != "general"}, calls
    assert len(bb.configs()) == 84 and bb.N_ITEMS == 8400 == 84 * bb.DATAPOINTS


# ---------------------------------------------------------------------------------------------------------------- the items
def test_items_are_identical_on_every_build_and_pinned(rows):
    raw = bb.dumps_rows(rows)
    assert raw == bb.dumps_rows(bb.build_items())                                                # generated twice: the same bytes
    assert hashlib.sha256(raw).hexdigest() == bb.ITEMS_SHA256 and len(rows) == 8400
    assert len({r["id"] for r in rows}) == 8400


def test_item_counts_and_shapes(rows):
    per = {}
    for r in rows:
        per.setdefault((r["task"], r["list_size"]), []).append(r)
    assert len(per) == 84 and all(len(v) == 100 for v in per.values())
    assert {(t, s) for (t, s) in per if t == "sum"} == {("sum", 8), ("sum", 16)}
    assert {s for (t, s) in per if t == "median"} == {9, 16} and {s for (t, s) in per if t == "multiplication"} == {2, 4}
    assert {s for (t, s) in per if t == "dot_product"} == {5, 10} and {s for (t, s) in per if t == "set_difference"} == {6, 12}
    assert {(t, s) for (t, s) in per if t in bb.NO_LIST_SIZE} == {("comparison", None), ("division", None), ("subtraction", None)} and ("absolute_difference", 2) in per
    for r in rows:
        dp = json.loads(r["data_point"])
        assert r["prompt"] == bb.make_task(r["task"], r["min_val"], r["max_val"]).create_prompt(dp)
        assert (r["min_val"], r["max_val"]) == {"element_frequency": (-10, 10), "set_intersection": (-20, 20), "set_difference": (-20, 20)}.get(r["task"], (-100, 100))
    lists = [json.loads(r["data_point"]) for r in per[("sorting", 8)]]
    assert all(len(l) == 8 and len(set(l)) == 8 and all(-100 <= x <= 100 for x in l) for l in lists)


@pytest.mark.parametrize("key", sorted(GOLDEN, key=str))
def test_golden_items(rows, key):
    task, size, i = key
    r = next(r for r in rows if (r["task"], r["list_size"], r["index"]) == key)
    assert (r["prompt"], r["gold"]) == GOLDEN[key]


def test_prompts_equal_the_stored_prompts_of_the_official_runs(rows):
    if not STORED.is_dir():
        pytest.skip("stored official runs not available")
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((r["task"], r["list_size"]), []).append(r)
    n = 0
    for model in ("LFM2-8B-A1B", "step-179000"):
        for tdir in glob.glob(f"{STORED}/{model}/*/task_results/*"):
            task = os.path.basename(tdir)
            sizes = sorted(k[1] for k in by_cfg if k[0] == task and k[1] is not None)
            for case in os.listdir(tdir):
                if not case.startswith("test_case_"):
                    continue
                suffix = case[len("test_case_"):]
                size = int(suffix) if suffix.isdigit() and int(suffix) in sizes else sizes[int(suffix)] if suffix.isdigit() else (2 if task == "absolute_difference" else None)
                stored = json.load(open(glob.glob(f"{tdir}/{case}/detailed_results_fold_*.json")[0]))
                assert [s["prompt"] for s in stored] == [m["prompt"] for m in by_cfg[(task, size)]], (model, task, size)
                n += len(stored)
    assert n == 16800


def test_stored_verdicts_equal_ours(rows):
    """The first 10 answers of every configuration of both stored official runs (1,680 answers) get the official verdict (accuracy, instruction_followed, empty)."""
    if not STORED.is_dir():
        pytest.skip("stored official runs not available")
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((r["task"], r["list_size"]), []).append(r)
    n = 0
    for model in ("LFM2-8B-A1B", "step-179000"):
        for tdir in glob.glob(f"{STORED}/{model}/*/task_results/*"):
            task = os.path.basename(tdir)
            sizes = sorted(k[1] for k in by_cfg if k[0] == task and k[1] is not None)
            for case in os.listdir(tdir):
                if not case.startswith("test_case_"):
                    continue
                suffix = case[len("test_case_"):]
                size = int(suffix) if suffix.isdigit() and int(suffix) in sizes else sizes[int(suffix)] if suffix.isdigit() else (2 if task == "absolute_difference" else None)
                stored = json.load(open(glob.glob(f"{tdir}/{case}/detailed_results_fold_*.json")[0]))
                for s, m in list(zip(stored, by_cfg[(task, size)]))[:10]:
                    v = bb.official_verdict(task, json.loads(m["data_point"]), s["model_response"])
                    assert v["accuracy"] == float(s["accuracy"]) and int(bool(v["instruction_followed"])) == int(bool(s["instruction_followed"])) and v["generation_success"] == s["generation_success"], (model, task, size, m["index"])
                    n += 1
    assert n == 1680


# ---------------------------------------------------------------------------------------------------------------- the scorer
def _item(rows, task, size=None, i=0):
    r = next(r for r in rows if r["task"] == task and r["index"] == i and (size is None or r["list_size"] == size))
    return json.loads(r["data_point"]), json.loads(r["gold"])


def test_score_gold_formats_and_wrong_answers(rows):
    dp, gold = _item(rows, "sum", 8)                                      # -190
    for resp in (f"\\boxed{{{gold}}}", f"so the sum is $\\boxed{{{gold}}}$.", f"The answer is {gold}.", f"{gold}", f"\\boxed{{ {gold} }}", f"\\boxed{{{gold}.0}}", f"Done.\n\\boxed{{{gold}}}\n\nGood luck, next question please"):
        assert bb.score_item("sum", dp, resp)["correct"] == 1.0, resp
    for resp in (f"\\boxed{{{gold + 1}}}", f"\\boxed{{{-gold}}}", "\\boxed{answer}", "I do not know.", "\\boxed{}", "nothing"):
        s = bb.score_item("sum", dp, resp)
        assert s["correct"] == 0.0 and s["empty"] == 0.0, resp
    lst, gl = _item(rows, "sorting", 8)
    assert bb.score_item("sorting", lst, "\\boxed{" + json.dumps(gl) + "}")["correct"] == 1.0
    assert bb.score_item("sorting", lst, "\\boxed{" + ", ".join(map(str, gl)) + "}")["correct"] == 1.0
    assert bb.score_item("sorting", lst, "\\boxed{" + json.dumps(gl[::-1]) + "}")["correct"] == 0.0           # order matters for sorting
    assert bb.score_item("sorting", lst, "\\boxed{" + json.dumps(gl[:-1]) + "}")["correct"] == 0.0
    assert bb.score_item("sorting", lst, "\\boxed{" + json.dumps(lst) + "}")["correct"] == 0.0 or lst == gl


def test_last_box_counts_for_numbers_and_lists_differ(rows):
    """The official number parsers take the LAST box (a base model that goes on with another box after the right answer is wrong); the list parsers of the per-task modules do not read it that way."""
    dp, gold = _item(rows, "sum", 8)
    assert bb.score_item("sum", dp, f"\\boxed{{{gold}}}\n\nNext problem.\n\\boxed{{4}}")["correct"] == 0.0
    assert bb.score_item("sum", dp, f"\\boxed{{4}}\n\n\\boxed{{{gold}}}")["correct"] == 1.0
    dp2, g2 = _item(rows, "reverse_list", 8)
    assert bb.score_item("reverse_list", dp2, f"\\boxed{{{json.dumps(g2)}}}\n\n\\boxed{{4}}")["correct"] == 1.0


def test_empty_and_whitespace_replies(rows):
    dp, _ = _item(rows, "sum", 8)
    e = bb.score_item("sum", dp, "")
    assert e["correct"] == 0.0 and e["empty"] == 1.0 and e["timeout"] == 0.0 and e["eval_error"] == 0.0 and e["official_correct"] == 0.0
    w = bb.score_item("sum", dp, "\n\n")                                                              # not empty for the official code (bool("\n\n") is True): a scored wrong answer
    assert w["empty"] == 0.0 and w["correct"] == 0.0 and w["instruction_followed"] == 0.0


def test_deadline_counts_wrong_and_is_not_swallowed(rows):
    dp, gold = _item(rows, "element_frequency", 16)
    slow = "[[-1,2]," + "[-1,2]," * 200                                                                # an unclosed [[ and 200 pairs: the official regex of parse_element_frequency_answer backtracks without end (the shape of the 3 stored answers that hit the 10 s deadline)
    t0 = time.time()
    s = bb.score_item("element_frequency", dp, slow, seconds=0.3)
    assert s["timeout"] == 1.0 and s["correct"] == 0.0 and s["eval_error"] == 0.0 and s["empty"] == 0.0 and time.time() - t0 < 5
    import signal
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)                                         # the timer and the handler are restored
    assert bb.score_item("element_frequency", dp, "\\boxed{" + json.dumps(gold) + "}")["correct"] == 1.0
    out = []
    def other_thread():                                                                                # off the main thread there is no deadline (no signal): it still scores
        out.append(bb.score_item("element_frequency", dp, "\\boxed{" + json.dumps(gold) + "}", seconds=0.3))
    import threading
    th = threading.Thread(target=other_thread)
    th.start(); th.join()
    assert out[0]["correct"] == 1.0


def test_second_minimum_official_bug_and_fix(rows):
    dp, gold = _item(rows, "second_minimum", 8)
    inst = bb.make_task("second_minimum")
    echo = f"Look at the list {dp}. The distinct values are ... \\boxed{{{gold}}}"
    off = bb.official_verdict("second_minimum", dp, echo)
    fixed = bb.official_verdict("second_minimum", dp, echo, fixed=True)
    assert off["accuracy"] == 0.0 and fixed["accuracy"] == 1.0                                         # the official code takes the first number of the reply (the first list element)
    s = bb.score_item("second_minimum", dp, echo)
    assert s["correct"] == 1.0 and s["official_correct"] == 0.0
    plain = f"\\boxed{{{gold}}}"                                                                       # a reply that is only the box: both agree
    assert bb.official_verdict("second_minimum", dp, plain)["accuracy"] == 1.0 == bb.official_verdict("second_minimum", dp, plain, fixed=True)["accuracy"]
    assert bb.score_item("sum", _item(rows, "sum", 8)[0], "\\boxed{-190}")["official_correct"] == bb.score_item("sum", _item(rows, "sum", 8)[0], "\\boxed{-190}")["correct"]    # every other task: no difference
    assert inst.evaluate_response("", dp)["ground_truth"] == gold


# ---------------------------------------------------------------------------------------------------------------- the task and the metric
def test_tasks_table():
    t = beyondbench.TASKS_TABLE
    assert [x.name for x in t] == [f"beyondbench_easy_base:{task}" for task in bb.TASKS]
    assert {x.generation_size for x in t} == {4096} and {tuple(x.stop_sequence) for x in t} == {()} and {x.hf_repo for x in t} == {"beyondbench_local"}
    assert {x.hf_revision for x in t} == {"0.2.1"} and [x.hf_subset for x in t] == list(bb.TASKS) and {tuple(x.evaluation_splits) for x in t} == {("train",)}
    m = t[0].metrics[0]
    assert list(m.metric_name) == list(beyondbench.NAMES) and set(m.corpus_level_fn) == set(beyondbench.NAMES)


def test_prompt_function_and_metric(rows):
    line = next(r for r in rows if r["task"] == "sum" and r["list_size"] == 8 and r["index"] == 0)
    doc = beyondbench.beyondbench_prompt(line, "beyondbench_easy_base:sum|0")
    assert doc.query == GOLDEN[("sum", 8, 0)][0] + beyondbench.CUE and doc.query.endswith("\n\nSolution:") and doc.choices == ["-190"] and doc.gold_index == 0 and doc.specific["task"] == "sum" and doc.specific["data_point"] == json.loads(line["data_point"])
    from lighteval.models.model_output import ModelResponse
    fn = beyondbench.beyondbench_metric.sample_level_fn
    ok = fn.compute(doc, ModelResponse(text=["so \\boxed{-190}"]))
    assert ok == {"acc": 1.0, "acc_official": 1.0, "instruction_followed": 1.0, "empty": 0.0, "timeout": 0.0, "eval_error": 0.0}
    assert doc.specific["verdict"] == "correct" and doc.specific["extracted_predictions"] == ["-190"] and doc.specific["extracted_golds"] == ["-190"]
    empty = fn.compute(doc, ModelResponse(text=[""]))
    assert empty["acc"] == 0.0 and math.isnan(empty["acc_official"]) and empty["empty"] == 1.0 and doc.specific["verdict"] == "empty"
    wrong = fn.compute(doc, ModelResponse(text=["\\boxed{5}"]))
    assert wrong["acc"] == 0.0 and wrong["acc_official"] == 0.0 and wrong["instruction_followed"] == 1.0 and doc.specific["verdict"] == "wrong"
    agg = beyondbench.beyondbench_metric.corpus_level_fn
    # 3 items: right, wrong, empty: acc counts the empty one (1/3), acc_official leaves it out (1/2), as the package's process_fold_metrics does
    assert agg["acc"]([1.0, 0.0, 0.0]) == pytest.approx(1 / 3) and agg["acc_official"]([1.0, 0.0, float("nan")]) == pytest.approx(0.5) and math.isnan(agg["acc_official"]([float("nan")]))
    assert "mean" not in agg["acc_official"].__name__ and "mean" in agg["acc"].__name__                # lighteval: plain stderr for "mean", bootstrap otherwise (NaN-safe)


def test_loader_checks_the_pin(rows, tmp_path, monkeypatch):
    monkeypatch.setenv("TITAN_EVAL_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="titan-eval prepare beyondbench"):
        bb.load_items("sum")
    d = tmp_path / "xdg" / bb.ITEMS_FOLDER
    d.mkdir(parents=True)
    raw = bb.dumps_rows(rows)
    (d / bb.ITEMS_FILE).write_bytes(raw)
    got = bb.load_items("sum")
    assert len(got) == 200 and {r["task"] for r in got} == {"sum"} and len(bb.load_items()) == 8400
    ds = beyondbench.load_beyondbench("median")
    assert list(ds) == ["train"] and len(ds["train"]) == 200 and ds["train"][0]["task"] == "median"
    with pytest.raises(KeyError):
        beyondbench.load_beyondbench("hard_task")
    (d / bb.ITEMS_FILE).write_bytes(raw.replace(b"Add the following", b"Add the follow1ng", 1))
    with pytest.raises(ValueError, match="is not the pinned"):
        bb.load_items("sum")
