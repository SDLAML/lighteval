"""Tests of the generation half of the judge-graded chat evals (tasks/tasks/chat_judged.py): SimpleQA, SimpleQA Verified, AA-Omniscience, AA-LCR, HLE (text only).

The grader prompts, the readers of the judge's reply and the metrics live in titan_eval (titan_eval/evals/judge_graders.py, tests/test_chat_judge.py there): the fork only builds the
prompts, puts `question` / `gold` into `Doc.specific` and counts `answered`.

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_judged.py`. Tests that read the AA-LCR documents skip when the pinned dataset is not in the Hugging Face cache.
"""

import re
import unittest  # noqa: F401  (pytest collects plain functions; keeps the file importable by unittest runners)

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks import chat_judged as cj


def test_four_tasks_with_pinned_revisions_and_no_judge_metric():
    assert [t.name for t in cj.TASKS_TABLE] == ["simpleqa_chat", "simpleqa_verified_chat", "aa_omniscience_chat", "aa_lcr_chat", "hle_chat"]
    for t in cj.TASKS_TABLE:
        assert re.fullmatch(r"[0-9a-f]{40}", t.hf_revision), t.name
        assert t.metrics[0] is cj.answered_metric and len(t.metrics) == 1, "the benchmark's metrics are the judge stage's, not lighteval's"
        assert t.stop_sequence == [] and t.generation_size == 32768 and t.few_shots_split is None
    assert cj.EXTRA_FILES == {"aa_lcr_chat": [("ArtificialAnalysis/AA-LCR", "extracted_text/AA-LCR_extracted-text.zip", cj.LCR_REVISION)]}


def test_simpleqa_prompt_is_the_bare_question():
    d = cj.simpleqa_chat_prompt({"problem": "Who wrote it?", "answer": "Me", "metadata": "{}"}, "simpleqa_chat|0")
    assert d.query == "Who wrote it?" and d.instruction is None and d.get_golds() == ["Me"]
    assert d.specific == {"question": "Who wrote it?", "gold": "Me"}


def test_simpleqa_verified_prompt_is_the_bare_question_and_keeps_the_acceptable_range():
    row = {"original_index": 5, "problem": "How many citations?", "answer": "120k (acceptable range: anything between 118k and 122k)", "topic": "Science and technology",
           "answer_type": "Number", "multi_step": False, "requires_reasoning": True, "urls": "http://x"}
    d = cj.simpleqa_verified_chat_prompt(row, "simpleqa_verified_chat|0")
    assert d.query == "How many citations?" and d.instruction is None and d.get_golds() == [row["answer"]]
    assert d.specific == {"question": "How many citations?", "gold": row["answer"], "original_index": 5, "topic": "Science and technology", "answer_type": "Number",
                          "multi_step": False, "requires_reasoning": True}
    t = next(t for t in cj.TASKS_TABLE if t.name == "simpleqa_verified_chat")
    assert (t.hf_repo, t.hf_subset, t.hf_revision, list(t.evaluation_splits)) == ("google/simpleqa-verified", "simpleqa_verified", "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc", ["eval"])


def test_omniscience_prompt_is_the_cards_answer_prompt_plus_the_question():
    row = {"domain": "Law", "topic": "Tort", "subtopic": "Negligence", "question_id": 7, "question": "Which case?", "answer": "Donoghue v Stevenson"}
    d = cj.aa_omniscience_chat_prompt(row, "x")
    assert d.query == ("You are answering questions about Law, and in particular Negligence.\nYou will be given a question, answer with JUST the answer (no explanation).\n"
                       "If you do not know the answer, or you need more context or tools to answer the question,\n"
                       "be clear about this - it is better that you say this than get the wrong answer.\n\nWhich case?")
    assert d.specific == {"question": "Which case?", "gold": "Donoghue v Stevenson", "domain": "Law", "topic": "Tort", "subtopic": "Negligence", "question_id": 7}


def test_hle_prompt_drops_image_questions_and_keeps_the_text():
    row = {"id": "a1", "question": "Compute it.", "image": "", "answer": "3", "answer_type": "exactMatch", "category": "Math", "raw_subject": "algebra"}
    d = cj.hle_chat_prompt(row, "hle_chat|0")
    assert d.query == "Compute it." and d.specific["gold"] == "3" and d.specific["category"] == "Math" and d.instruction is None
    assert cj.hle_chat_prompt({**row, "image": "data:image/png;base64,AAAA"}, "x") is None
    assert cj.hle_chat_prompt({**row, "image": None}, "x") is not None


def test_lcr_prompt_template_and_document_order():
    p = cj.lcr_prompt("Which one?", ["first text", "second text"])
    assert p == ("BEGIN INPUT DOCUMENTS\n\nBEGIN DOCUMENT 1:\nfirst text\nEND DOCUMENT 1\n\nBEGIN DOCUMENT 2:\nsecond text\nEND DOCUMENT 2\n\nEND INPUT DOCUMENTS\n\n"
                 "Answer the following question using the input documents provided above.\n\nSTART QUESTION\n\nWhich one?\n\nEND QUESTION\n")


def test_lcr_documents_resolve_odd_file_names_from_the_cached_zip():
    try:
        docs = cj.lcr_documents("Marketing", "mkt_gaming", ["13817.txt", "402813954_17. 260-275 Sinem Eyice Başev.txt"])
    except Exception as e:          # noqa: BLE001  (not in the cache: login node `titan-eval prefetch --task aa_lcr_chat|0`)
        print(f"skip: AA-LCR zip not in the cache ({type(e).__name__})")
        return
    assert len(docs) == 2 and all(len(x) > 1000 for x in docs)
    try:
        cj.lcr_documents("Marketing", "mkt_gaming", ["missing.txt"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing document must be an error")


def test_answered_counts_non_empty_scored_text():
    f = cj.answered_metric.sample_level_fn.compute
    assert f(None, ModelResponse(text=["a"], text_post_processed=["a"])) == {"answered": 1.0}
    assert f(None, ModelResponse(text=["a", "b"], text_post_processed=["", "  \n"])) == {"answered": 0.0}
    assert f(None, ModelResponse(text=["a", "b"], text_post_processed=["x", ""])) == {"answered": 0.5}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
