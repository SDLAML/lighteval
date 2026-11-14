"""Chat PolyMath and MGSM (tasks/tasks/chat_polymath.py): polymath_chat (17 non-English languages x 4 levels, full set), polymath_en_chat (English, 4 levels),
mgsm_chat (40 languages, no English).

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_polymath.py`. No GPU, no dataset read: the task table and the metric only
(the opt-in subsets are tested in test_sampling.py; the unscorable-gold count with the pinned data at the end, skipped when the dataset is not cached).
"""

import unittest
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
    assert len(poly) == 68, "17 languages x 4 levels"
    assert Counter(n.split(":")[1] for n in poly) == {"low": 17, "medium": 17, "high": 17, "top": 17}
    langs = Counter(n.split(":")[2] for n in poly)
    assert langs == {lang: 4 for lang in cp.POLYMATH_LANGUAGES if lang != "en"} and "en" not in langs, "English is polymath_en_chat"
    en = [n for n in names if n.startswith("polymath_en_chat:")]
    assert en == [f"polymath_en_chat:{lvl}" for lvl in cp.POLYMATH_LEVELS]
    mgsm = [n for n in names if n.startswith("mgsm_chat:")]
    assert len(mgsm) == 40 and "mgsm_chat:en" not in mgsm and "mgsm_chat:zh" in mgsm
    assert all(c.hf_filter is None for c in cp.TASKS_TABLE), "the full sets: no subset filter without data.sample_per_group"


def test_en_chat_reads_the_english_config_with_the_english_note():
    for lvl in cp.POLYMATH_LEVELS:
        c = _cfg(f"polymath_en_chat:{lvl}")
        assert c.hf_repo == "Qwen/PolyMath" and c.hf_subset == "en" and list(c.evaluation_splits) == [lvl]
        d = c.prompt_function({"id": f"{lvl}-en-0", "question": "How many?", "answer": "18"}, "t")
        assert d.query.endswith(cp.POLYMATH_NOTE["en"]) and d.specific["language"] == "en" and d.specific["level"] == lvl
    a, b = _cfg("polymath_en_chat:top"), _cfg("polymath_chat:top:de")
    assert a.metrics[0].metric_name == b.metrics[0].metric_name, "same metrics as polymath_chat"


def test_revisions_and_repos_are_the_pins():
    assert all(c.hf_repo == "Qwen/PolyMath" and c.hf_revision == "71e0db902e0ece05e208dfd8b1695bd3d95cf130"
               for c in cp.TASKS_TABLE if c.name.startswith("polymath"))
    assert all(c.hf_repo == "CohereLabs/global-mgsm" and c.hf_revision == "29087245a1a9788d4e0413f4927e994d49633bbb"
               for c in cp.TASKS_TABLE if c.name.startswith("mgsm"))
    assert all(list(c.hf_avail_splits) == [c.name.split(":")[1]] for c in cp.TASKS_TABLE if c.name.startswith("polymath"))


def test_polymath_prompt_is_the_question_plus_the_official_note():
    line = {"id": "low-de-3", "question": "  Wie viele?  ", "answer": "18"}
    doc = _cfg("polymath_chat:low:de").prompt_function(line, "t")
    assert doc.query == "Wie viele?\n\nHinweis: Bitte setzen Sie die endgültige Antwort in $\\boxed{}$."
    doc = _cfg("polymath_en_chat:top").prompt_function({"id": "top-en-0", "question": "How many?", "answer": "18"}, "t")
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
    assert set(m.sample_level_fn.names) == {"avg@n", "avg@n_lenient", "dw_acc", "gold_unscorable"}
    assert m.higher_is_better["gold_unscorable"] is False and m.higher_is_better["avg@n"] is True


def test_mgsm_prompt_and_scoring():
    doc = _cfg("mgsm_chat:sw").prompt_function({"question": "Ni kiasi gani?", "answer": "18"}, "t")
    assert doc.query == MATH_PROMPT.format(problem="Ni kiasi gani?") and doc.choices == ["$18$"]
    assert chat_math_metric(1).sample_level_fn.compute(doc, ModelResponse(text=["x \\boxed{18}"], text_post_processed=["x \\boxed{18}"]))["avg@n"] == 1.0
    assert _cfg("mgsm_chat:sw").metrics[0].sample_level_fn.num_samples() == 1


def test_unscorable_golds_are_counted_not_removed():
    ok = {"avg@n": 1.0}
    assert cp.gold_is_scorable("18") and cp.gold_is_scorable("$\\frac{1}{2}$")
    for bad in ("$2^{2^l}$ for $l=1,2,4,8$", "$P(x)=c(x^2+3)$ for any constant $c$", "$b \\geq $-\\frac{2}{3}$", "$\\frac{1}{2\\sqrt{2} \\ln\\frac{\\sqrt{2}+2}{2-\\sqrt{2}}$"):     # the real golds of top-en-50, top-en-88, medium-en-1, medium-en-111
        assert not cp.gold_is_scorable(bad), bad
    good = _score("low", "\\boxed{18}")
    assert good["avg@n"] == 1.0 and good["gold_unscorable"] == 0.0
    bad_doc = cp.Doc(query="", choices=[cp._bare_gold("$2^{2^l}$ for $l=1,2,4,8$")], gold_index=0, specific={"level": "top"})
    r = cp.polymath_metric(1).sample_level_fn.compute(bad_doc, ModelResponse(text=["\\boxed{2^{2^l}}"], text_post_processed=["\\boxed{2^{2^l}}"]))
    assert r["gold_unscorable"] == 1.0 and r["avg@n"] == 0.0, "the item stays in the set, scores 0 and is counted"


def test_unscorable_count_with_the_pinned_data():
    """90 of the 9,000 golds (5 problems x 18 languages: 2 medium + 3 top each), 85 of the 8,500 non-English ones, 5 of the 500 English ones."""
    from lighteval.tasks.lighteval_task import load_dataset_pinned

    n = Counter()
    try:
        for lang in cp.POLYMATH_LANGUAGES:
            for lvl in cp.POLYMATH_LEVELS:
                ds = load_dataset_pinned(cp.POLYMATH_REPO, lang, revision=cp.POLYMATH_REVISION, split=lvl)
                assert len(ds) == cp.PER_CELL == 125, (lang, lvl)
                n[lang] += sum(1 for a in ds["answer"] if not cp.gold_is_scorable(a))
    except Exception as e:
        if isinstance(e, AssertionError):
            raise
        raise unittest.SkipTest(f"Qwen/PolyMath not cached: {type(e).__name__}")
    assert sum(n.values()) == 90 and set(n.values()) == {5} and n["en"] == 5 and sum(v for k, v in n.items() if k != "en") == 85


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok", name)
            except unittest.SkipTest as e:
                print("skip", name, e)
