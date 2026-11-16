"""Tests of the chat reasoning / knowledge twins (src/lighteval/tasks/tasks/chat_reasoning.py): mmlu_chat and gpqa_main_chat.

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_reasoning.py`. No GPU; the dataset test skips when the
pinned data is not in the Hugging Face cache.
"""

import re

from lighteval.metrics.chat_metrics import chat_choice_metric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_reasoning
from lighteval.tasks.tasks.mmlu import _MMLU_SUBSETS


def _resp(*texts):
    return ModelResponse(text=list(texts), text_post_processed=list(texts))


def _mc_doc(gold=2, n=4):
    return Doc(query="q", choices=list("ABCDEFGHIJ")[:n], gold_index=gold, task_name="t")


def test_task_table_names_and_counts():
    names = [t.name for t in chat_reasoning.TASKS_TABLE]
    assert len(names) == 57 + 1 + 9 + 27 and "gpqa_main_chat" in names
    assert names[0] == f"mmlu_chat:{_MMLU_SUBSETS[0]}" and sum(n.startswith("agieval_en_chat:") for n in names) == 9
    for t in chat_reasoning.TASKS_TABLE:
        assert t.generation_size == 32768 and t.stop_sequence == [] and t.few_shots_split is None
        assert t.version == 1, t.name           # 11-16: every task of this module changed (A1 / A2 / A4)


def test_mmlu_chat_prompt_is_the_chat_mc_prompt_with_int_gold():
    line = {"question": "Which?", "choices": ["w", "x", "y", "z"], "answer": 2}
    d = chat_reasoning.mmlu_chat_prompt(line, "t", subject="anatomy")
    assert d.gold_index == 2 and d.choices == ["A", "B", "C", "D"]
    assert "where LETTER is one of ABCD." in d.query and "Which?\n\nA) w\nB) x\nC) y\nD) z" in d.query
    assert d.specific == {"subject": "anatomy"} and "'Answer: $LETTER'" in d.query


def test_mmlu_chat_prompt_accepts_letter_gold():
    line = {"question": "Q", "choices": ["a", "b", "c", "d"], "answer": "D"}
    assert chat_reasoning.mmlu_chat_prompt(line, "t", subject="x").gold_index == 3


def test_scoring_is_the_chat_choice_metric():
    m = chat_choice_metric(1).sample_level_fn
    assert m.compute(_mc_doc(), _resp("Reasoning ... Answer: C"))["acc"] == 1.0
    assert m.compute(_mc_doc(), _resp("Answer: B"))["acc"] == 0.0
    assert m.compute(_mc_doc(), _resp("Answer: B or C"))["acc"] == 0.0          # a hedge is no answer
    assert m.compute(_mc_doc(), _resp("C is probably right"))["acc"] == 0.0     # unanchored: diagnostic only
    assert m.compute(_mc_doc(), _resp("C is probably right"))["acc_unanchored"] == 1.0
    assert m.compute(_mc_doc(), _resp(""))["acc"] == 0.0


def test_gpqa_main_chat_is_the_diamond_prompt_on_the_main_subset():
    t = next(t for t in chat_reasoning.TASKS_TABLE if t.name == "gpqa_main_chat")
    assert t.hf_subset == "gpqa_main" and t.hf_repo == "Idavidrein/gpqa"
    assert t.hf_revision == "83022cefff930aea54f654c0b282e74b9eeda5c6"
    assert t.metrics[0].sample_level_fn.n == 8 and tuple(t.evaluation_splits) == ("train",)


def _bbh_doc(subset, target, input_text):
    return chat_reasoning.bbh_chat_prompt({"input": input_text, "target": target}, "t", subset=subset)


def test_bbh_text_answers_are_exact_after_normalisation():
    d = _bbh_doc("web_of_lies", "Yes", "Question: Mal lies. Does Mal tell the truth?")
    score = chat_reasoning._bbh_score
    assert score(d, "Work.\n\nAnswer: Yes.")["em"] == 1.0
    assert score(d, "Answer: **yes**")["em"] == 1.0
    assert score(d, "Answer: No")["em"] == 0.0
    assert score(d, "Answer: Yes or No")["em"] == 0.0            # a hedge is not the answer
    assert score(d, "Yes, Mal tells the truth.")["em"] == 0.0    # no anchor
    assert score(d, "")["em"] == 0.0


def test_bbh_letter_answers_use_the_option_count_up_to_r():
    options = "\n".join(f"({c}) x{c}" for c in "ABCDEFGHIJKLMNOPQR")
    d = _bbh_doc("reasoning_about_colored_objects", "(K)", f"Q?\nOptions:\n{options}")
    assert len(d.specific["n_options"]) == 18
    assert chat_reasoning._bbh_score(d, "Answer: (K)")["em"] == 1.0
    assert chat_reasoning._bbh_score(d, "Answer: (R)")["em"] == 0.0
    assert chat_reasoning._bbh_score(d, "Answer: (K) or (L)")["em"] == 0.0
    small = _bbh_doc("tracking_shuffled_objects_three_objects", "(B)", "Q?\n(A) x\n(B) y\n(C) z")
    assert chat_reasoning._bbh_score(small, "Answer: (D)")["em"] == 0.0   # a letter beyond the 3 options


def test_bbh_task_table_is_27_subsets_with_the_pin():
    names = [t.name for t in chat_reasoning.TASKS_TABLE if t.name.startswith("bbh_cot_chat:")]
    assert len(names) == 27 and all(t.hf_revision == "982bb89fd79532a8ac676a61fc42eb1aeec63f99" for t in chat_reasoning.TASKS_TABLE if t.name in names)


def test_bbh_text_answers_strip_latex_wrappers_and_read_the_official_phrase():
    """A1 (owner 11-16): OLMES bbh.py l.246-257 wrappers are stripped; "So the answer is X" (the official BBH phrase) is read without a colon when no "Answer:" line exists."""
    d = _bbh_doc("multistep_arithmetic_two", "-35", "Q?")
    for ok in ("Answer: \\(-35\\)", "Answer: $-35$", "Answer: \\[ -35 \\]", "Answer: **-35**", "Answer: (-35)", 'Answer: "-35"', "Answer: `-35`", "Answer: -35.", "So the answer is -35.",
               "Thus, the answer is \\(-35\\).", "answer is: -35", "Answer: \\(-35\\)\n\nThe answer is that I am done"):
        assert chat_reasoning._bbh_score(d, ok)["em"] == 1.0, ok
    for bad in ("Answer: -36", "The answer is -36.", "So the answer is -35 because of the sign", "no answer line -35", "To answer this question we compute -35", ""):
        assert chat_reasoning._bbh_score(d, bad)["em"] == 0.0, bad
    assert chat_reasoning._bbh_score(d, "Answer: -36\nSo the answer is -35.")["em"] == 0.0, "the colon line has priority: a response that has one scores exactly as before"
    yn = _bbh_doc("navigate", "No", "Q?")
    assert chat_reasoning._bbh_score(yn, "So the answer is no.")["em"] == 1.0 and chat_reasoning._bbh_score(yn, "**Answer: No**")["em"] == 1.0
    dyck = _bbh_doc("dyck_languages", "] )", "Q?")
    assert chat_reasoning._bbh_score(dyck, "Answer: ] )")["em"] == 1.0 and chat_reasoning._bbh_score(dyck, "Answer: ( ] ) ")["em"] == 0.0, "the (..) pair is not a wrapper of a bracket answer"
    # letter subsets: the markup the simple-evals normalisation removes
    letter = _bbh_doc("snarks", "(B)", "Q?\n(A) x\n(B) y")
    for ok in ("**Answer:** **(B)**", "Answer: \\(\\boxed{B}\\)", "So the answer is (B).", "Answer: B"):
        assert chat_reasoning._bbh_score(letter, ok)["em"] == 1.0, ok
    # everything the old normalisation read stays read: new(x) is a function of old(x)
    old = lambda x: re.sub(r"\s+", " ", x.replace("**", "").replace("`", "").strip().strip("$").strip()).rstrip(".").strip().lower()   # noqa: E731
    for x in ["True", "true.", "**True**", "`yes`", "$12$", " 7 ", "a  b", "word"]:
        assert chat_reasoning._norm_text(x) == old(x), x


def test_agieval_has_the_nine_official_english_files_and_the_chat_twin_reads_them():
    from lighteval.tasks.tasks import agieval
    assert agieval.AGIEVAL_SUBSETS == ["aqua_rat", "gaokao-english", "logiqa-en", "lsat-ar", "lsat-lr", "lsat-rc", "sat-en", "sat-en-without-passage", "sat-math"]
    assert sorted(agieval.AGIEVAL_FILES) == sorted(agieval.agieval_file(x) for x in agieval.AGIEVAL_SUBSETS) and sum(v[0] for v in agieval.AGIEVAL_FILES.values()) == 2852
    assert agieval.AGIEVAL_COMMIT == "84ab72d94318290aad2e4ec820d535a95a1f7552"
    for t in chat_reasoning.TASKS_TABLE:
        if t.name.startswith("agieval_en_chat:"):
            assert (t.hf_repo, t.hf_revision, t.hf_subset) == (agieval.AGIEVAL_REPO, agieval.AGIEVAL_COMMIT, agieval.agieval_file(t.name.split(":")[1])) and t.version == 1
    for t in agieval.TASKS_TABLE:
        if t.name.endswith((":mcf", ":mcf_em")):
            assert t.hf_repo == agieval.AGIEVAL_REPO and t.hf_revision == agieval.AGIEVAL_COMMIT, t.name
    assert sum(t.name.endswith(":mcf_em") for t in agieval.TASKS_TABLE) == 9 and {t.version for t in agieval.TASKS_TABLE if t.name.endswith(":mcf_em")} == {3}
    from lighteval.tasks.tasks.agieval_demos import AGIEVAL_DEMOS
    assert {k: len(v) for k, v in AGIEVAL_DEMOS.items()}["gaokao-english"] == 3 and len(AGIEVAL_DEMOS["sat-en-without-passage"]) == 3


def test_agieval_sat_conflict_rows_are_dropped_in_both_sat_files():
    from lighteval.tasks.tasks import agieval
    opts = ["(A)reserved around unfamiliar people.", "(B)attuned to her immediate surroundings.", "(C)sympathetic to the needs of others.", "(D)anxious about her responsibilities."]
    q = "Throughout the passage, the narrator is portrayed as someone who is"
    passage = "At last, Old Widow Lau was done haggling with the driver and we stepped inside Father's shop. More text."
    for label in "AB":
        assert agieval.agieval_mcf_prompt({"passage": passage, "question": q, "options": opts, "label": label}, "t") is None, "sat-en"
        assert agieval.agieval_mcf_prompt({"passage": "", "question": q, "options": opts, "label": label}, "t") is None, "sat-en-without-passage (same two rows, no passage)"
        assert chat_reasoning.agieval_en_chat_prompt({"passage": None, "question": q, "options": opts, "label": label}, "t") is None
    other = {"passage": "", "question": "Which is the narrator's age?", "options": opts, "label": "A"}
    assert agieval.agieval_mcf_prompt(other, "t") is not None, "another question with the same options is kept"
    assert agieval.agieval_mcf_prompt({"passage": "Another passage.", "question": q, "options": opts, "label": "A"}, "t") is not None, "the same question after another passage is kept"


def test_agieval_official_files_load_and_count(tmp=None):
    """The nine pinned files (skipped when <cache>/xdg/agieval_v1 is not prepared): the row counts of the paper's task sizes (2,852 rows), 2,848 scored after the 4 conflicting SAT rows."""
    from lighteval.tasks.tasks import agieval
    if not (agieval.agieval_dir() / "lsat-ar.jsonl").is_file():
        print("skip: AGIEval files not prepared (titan-eval prepare agieval)")
        return
    kept = {}
    for sub in agieval.AGIEVAL_SUBSETS:
        ds = agieval.load_agieval(sub)["train"]
        assert len(ds) == agieval.AGIEVAL_FILES[agieval.agieval_file(sub)][0], sub
        kept[sub] = sum(chat_reasoning.agieval_en_chat_prompt(r, "t") is not None for r in ds)
    assert kept["sat-en"] == 204 and kept["sat-en-without-passage"] == 204 and kept["gaokao-english"] == 306 and sum(kept.values()) == 2848, kept


def test_agieval_chat_prompt_is_a_chat_mc_prompt_and_drops_the_duplicate_rows():
    line = {"question": "Which?", "options": ["(A) one", "(B) two", "(C) three", "(D) four"], "label": "C", "passage": None}
    d = chat_reasoning.agieval_en_chat_prompt(line, "t")
    assert d.gold_index == 2 and d.choices == ["A", "B", "C", "D"]
    assert "A) one" in d.query and "Question: Which?" in d.query and "one of ABCD." in d.query
    dup = {"question": "At last, Old Widow Lau was done haggling", "options": ["(A) reserved around unfamiliar people.", "(B) attuned to her immediate surroundings.", "(C) sympathetic to the needs of others.", "(D) anxious about her responsibilities."], "label": "A", "passage": "At last, Old Widow Lau was done haggling with the driver and we stepped inside Father's shop."}
    dup["question"] = "Throughout the passage, the narrator is portrayed as someone who is"
    assert chat_reasoning.agieval_en_chat_prompt(dup, "t") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
