"""Chat PolyMath and MGSM (tasks/tasks/chat_polymath.py): polymath_chat (18 languages x 4 levels, seeded 25 per cell), mgsm_chat (40 languages).

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_polymath.py`. No GPU, no dataset read: the subset file, the task
table and the metric only.
"""

import json
from collections import Counter

from lighteval.metrics.chat_metrics import chat_math_metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_polymath as cp
from lighteval.tasks.tasks.chat_evals import MATH_PROMPT


def _cfg(name):
    return next(c for c in cp.TASKS_TABLE if c.name == name)


def _doc(level, gold="18", language="en"):
    return Doc(query="q", choices=[gold], gold_index=0, specific={"id": f"{level}-{language}-0", "language": language, "level": level})


def _score(level, text, gold="18"):
    return cp.polymath_metric(1).sample_level_fn.compute(_doc(level, gold), ModelResponse(text=[text], text_post_processed=[text]))


def test_task_table_names_and_counts():
    names = [c.name for c in cp.TASKS_TABLE]
    assert len(names) == len(set(names)) == 112
    poly = [n for n in names if n.startswith("polymath_chat:")]
    assert len(poly) == 72
    assert Counter(n.split(":")[1] for n in poly) == {"low": 18, "medium": 18, "high": 18, "top": 18}
    assert Counter(n.split(":")[2] for n in poly) == {lang: 4 for lang in cp.POLYMATH_LANGUAGES}
    mgsm = [n for n in names if n.startswith("mgsm_chat:")]
    assert len(mgsm) == 40 and "mgsm_chat:en" not in mgsm and "mgsm_chat:zh" in mgsm


def test_revisions_and_repos_are_the_pins():
    assert all(c.hf_repo == "Qwen/PolyMath" and c.hf_revision == "71e0db902e0ece05e208dfd8b1695bd3d95cf130"
               for c in cp.TASKS_TABLE if c.name.startswith("polymath"))
    assert all(c.hf_repo == "CohereLabs/global-mgsm" and c.hf_revision == "29087245a1a9788d4e0413f4927e994d49633bbb"
               for c in cp.TASKS_TABLE if c.name.startswith("mgsm"))
    assert all(list(c.hf_avail_splits) == [c.name.split(":")[1]] for c in cp.TASKS_TABLE if c.name.startswith("polymath"))


def test_subset_file_is_25_per_cell_and_ids_match_their_cell():
    data = json.loads(cp.SUBSETS_FILE.read_text())
    assert data["seed"] == "chat_polymath-v1" and data["per_cell"] == 25
    cells = data["polymath"]
    assert sorted(cells) == sorted(cp.POLYMATH_LANGUAGES)
    all_ids = []
    for lang, levels in cells.items():
        assert sorted(levels) == sorted(cp.POLYMATH_LEVELS)
        for level, ids in levels.items():
            assert len(ids) == len(set(ids)) == 25
            assert all(i.startswith(f"{level}-{lang}-") for i in ids), (lang, level)
            all_ids += ids
    assert len(all_ids) == len(set(all_ids)) == 1800


def test_polymath_prompt_is_the_question_plus_the_official_note():
    line = {"id": "low-de-3", "question": "  Wie viele?  ", "answer": "18"}
    doc = _cfg("polymath_chat:low:de").prompt_function(line, "t")
    assert doc.query == "Wie viele?\n\nHinweis: Bitte setzen Sie die endgültige Antwort in $\\boxed{}$."
    doc = _cfg("polymath_chat:top:en").prompt_function({"id": "top-en-0", "question": "How many?", "answer": "18"}, "t")
    assert doc.query == "How many?\n\nNote: Please put the final answer in the $\\boxed{}$."
    assert doc.choices == ["$18$"] and doc.gold_index == 0


def test_gold_wrapping():
    assert cp._bare_gold("18") == "$18$"
    assert cp._bare_gold("$\\frac{1}{2}$") == "$\\frac{1}{2}$"
    assert cp._bare_gold("$2^{2^l}$ for $l=1,2,4,8$") == "$2^{2^l}$ for $l=1,2,4,8$"


def test_polymath_scores_strict_and_lenient():
    assert _score("low", "so \\boxed{18}")["avg@n"] == 1.0
    assert _score("low", "the answer is 18")["avg@n"] == 0.0, "strict: a box is required"
    assert _score("low", "the answer is 18")["avg@n_lenient"] == 1.0, "lenient: the last expression"
    assert _score("low", "\\boxed{17}")["avg@n"] == 0.0
    assert _score("low", "")["avg@n"] == 0.0


def test_dw_acc_is_the_official_weighted_average_on_a_balanced_set():
    # one correct item per level: the mean of the per-item dw_acc is (1 + 2 + 4 + 8) / 15 = 1.0 (all correct) and a single level's
    # correct item counts its weight: low 1/15, medium 2/15, high 4/15, top 8/15 of the balanced mean over four items
    per_level = {lvl: _score(lvl, "\\boxed{18}")["dw_acc"] for lvl in cp.POLYMATH_LEVELS}
    assert abs(per_level["low"] - 4 / 15) < 1e-9 and abs(per_level["top"] - 32 / 15) < 1e-9
    assert abs(sum(per_level.values()) / 4 - 1.0) < 1e-9, "all four correct = 1.0"
    assert abs(sum(_score(l, "no box")["dw_acc"] for l in cp.POLYMATH_LEVELS) / 4) < 1e-9
    top_only = [_score(l, "\\boxed{18}" if l == "top" else "\\boxed{0}")["dw_acc"] for l in cp.POLYMATH_LEVELS]
    assert abs(sum(top_only) / 4 - 8 / 15) < 1e-9, "only the top level correct = 8/15"


def test_polymath_metric_names_and_samples():
    m = _cfg("polymath_chat:high:zh").metrics[0]
    assert m.sample_level_fn.num_samples() == cp.N_POLY == 4
    assert m.sample_level_fn.primary == "avg@n"
    assert set(m.sample_level_fn.names) == {"avg@n", "avg@n_lenient", "dw_acc"}


def test_mgsm_prompt_and_scoring():
    doc = _cfg("mgsm_chat:sw").prompt_function({"question": "Ni kiasi gani?", "answer": "18"}, "t")
    assert doc.query == MATH_PROMPT.format(problem="Ni kiasi gani?") and doc.choices == ["$18$"]
    assert chat_math_metric(1).sample_level_fn.compute(doc, ModelResponse(text=["x \\boxed{18}"], text_post_processed=["x \\boxed{18}"]))["avg@n"] == 1.0
    assert _cfg("mgsm_chat:sw").metrics[0].sample_level_fn.num_samples() == 1


def test_filter_keeps_only_the_stored_ids():
    keep = cp.polymath_filter(frozenset({"low-en-0"}))
    assert keep({"id": "low-en-0"}) and not keep({"id": "low-en-1"})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
