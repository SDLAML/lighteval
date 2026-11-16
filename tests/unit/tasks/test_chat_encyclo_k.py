"""Tests of encyclo_k_chat (src/lighteval/tasks/tasks/chat_encyclo_k.py): the official prompt, the official answer extraction (eval/eval.py of
github.com/multimodal-art-projection/Encyclo-K @ 5549c738), the metric and its per-discipline breakdown.

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_encyclo_k.py`. No GPU; the dataset test skips when the pinned data is not in the
Hugging Face cache. The expected answers of `test_official_extraction_cases` are what the OFFICIAL eval.py returns (run on 11-16 with the official functions
loaded from the file; the port also gave the same answer as the official code on 46,943 stored chat generations of MMLU-Pro, 0 differences).
"""

import math
import time

from lighteval.metrics.chat_metrics import anchored_letter
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_encyclo_k as enc
from lighteval.tasks.tasks.encyclo_k_base import REVISION


OPTS = ["iv.vii.ii.vi.", "ii.viii.", "vi.iii.", "i.viii.", "v.ix.x."]


def _doc(gold="B", n=5, discipline="Law", options=None):
    letters = "ABCDEFGH"[:n]
    return Doc(task_name="encyclo_k_chat", query="q", choices=list(letters), gold_index=letters.index(gold),
               specific={"discipline": discipline, "options": list(options or OPTS[:n]), "uuid": "u"})


def _resp(*texts):
    return ModelResponse(text=list(texts), text_post_processed=list(texts))


# The evaluated official patterns (eval.py l.45, 50, 56 and l.82, 84, 86; `\r\n` of the third pattern are real CR / LF characters there, the same class).
OFFICIAL_LABEL_PATTERNS = (
    r"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([ABCDEFGHIJ])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"(?i:Answer)[\*\s]*:\s*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([ABCDEFGHIJ])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    "^[^\\w\r\n]*(?:[\\*\\$\\{(\\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){)?)*\\s*([ABCDEFGHIJ])(?:\\\\?\\}?\\$?\\)?\\]?\\}?)*(?:[\\s:\\.\\*)]|$)",
)
OFFICIAL_CONTENT_PATTERNS = (
    r"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(i\.ii\.|iii\.)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"(?i:Answer)\s*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(i\.ii\.|iii\.)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    "^[^\\w\r\n]*(?:[\\*\\$\\{\\(\\[\\\\(]*?(?:(?:\\\\boxed|\\\\mathbf|\\\\mathrm|\\\\text){)?)*\\s*(i\\.ii\\.|iii\\.)(?:\\\\?\\}?\\$?\\)?\\]?\\}?)*(?:[\\s:\\.\\*)]|$)",
)


def _same_regex(a: str, b: str) -> bool:
    """The CR / LF characters of the official third pattern and the `\\r\\n` escapes of ours are the same character class."""
    return a.replace("\r", "\\r").replace("\n", "\\n") == b.replace("\r", "\\r").replace("\n", "\\n")


def test_patterns_are_the_official_ones():
    assert len(enc.label_patterns()) == 3 and len(enc.content_patterns(["i.ii.", "iii."])) == 3
    for ours, official in zip(enc.label_patterns(), OFFICIAL_LABEL_PATTERNS):
        assert _same_regex(ours, official), (ours, official)
    for ours, official in zip(enc.content_patterns(["i.ii.", "iii."]), OFFICIAL_CONTENT_PATTERNS):
        assert _same_regex(ours, official), (ours, official)
    assert enc.OFFICIAL_LETTERS == "ABCDEFGHIJ"        # the official call is extract_option_labels(.., 'ABCDEFGHIJ'): always A-J, whatever the option count


# (name, generation, the letter the OFFICIAL eval.py extracts, the strict anchored reader of the other chat tasks)
CASES = [
    ("plain", "Analysis of each statement...\n\nAnswer: B", "B", "B"),
    ("bold word", "Analysis...\n\n**Answer: B**", "B", "B"),
    ("bold word then letter", "Analysis...\n\n**Answer:** B", "B", "B"),
    ("bold letter", "Analysis...\n\nAnswer: **B**", "B", "B"),
    ("boxed in dollars", "Analysis...\n\nAnswer: $\\boxed{B}$", "B", "B"),
    ("boxed sentence", "Analysis...\n\nThe answer is \\boxed{C}.", "C", "C"),
    ("paren letter", "Therefore, the correct answer is (D).", "D", "D"),
    ("final answer bold", "Statements i, iii are true.\n\n**Final Answer: C**", "C", "C"),
    ("letter with text", "Answer: B) ii.viii.", "B", "B"),
    ("hedge or: the official reads the first letter, the strict reader none", "Both look plausible.\n\nAnswer: B or C", "B", None),
    ("hedge slash: no match in both", "Answer: A/B", None, None),
    ("bare letter on the last line (pattern 3)", "Reasoning...\n\nC", "C", None),
    ("bare parenthesised letter", "Reasoning...\n\n(A)", "A", None),
    ("bare letter with a dot", "Reasoning...\n\nD.", "D", None),
    ("a last line that starts with the article A is read as the letter A", "Reasoning...\n\nA careful reading shows statements i and ii are correct.", "A", None),
    ("a last line that starts with the pronoun I is read as the letter I", "I cannot determine the answer from these statements.", "I", None),
    ("lower-case letter: extracted (IGNORECASE), then WRONG, not a miss", "answer: b", "b", None),
    ("a letter beyond the options is extracted (A-J), then wrong", "Answer: H", "H", None),
    ("answer in the text, last line without one: whole-text search", "Answer: C\n\nSome remarks that have no letter in them at all.", "C", "C"),
    ("two answers, last line without one: the FIRST in the text wins", "Answer: A\nOn reflection that was wrong.\nAnswer: B\nHope this helps.", "A", "B"),
    ("two answers, the last line has one: the last line wins", "Answer: A\nOn reflection that was wrong.\nAnswer: B", "B", "B"),
    ("option text alone on the last line (fallback)", "Reasoning.\n\niv.vii.ii.vi.", "A", None),
    ("option text after 'Answer:' (a colon is not allowed by the fallback)", "Reasoning.\n\nAnswer: vi.iii.", None, None),
    ("no answer at all", "Statements are hard to judge, so please consult a textbook on this topic.", None, None),
    ("Chinese answer line with a full-width colon", "分析如下。\n\n答案：B", None, None),
    ("the correct option is (the strict reader only knows 'answer')", "Hence, the correct option is E.", "E", None),
    ("trailing blanks", "Answer: C  \n\n  ", "C", "C"),
    ("markdown heading", "### Answer: D", "D", "D"),
    ("latex text", "Answer: \\text{B}", "B", "B"),
]


def test_official_extraction_cases():
    for name, text, official, strict in CASES:
        assert enc.official_answer(text, OPTS) == official, (name, enc.official_answer(text, OPTS))
        assert anchored_letter(text, 5) == strict, (name, anchored_letter(text, 5))


def test_score_acc_miss_and_the_strict_diagnostic():
    s = enc._score(_doc("B"), "Reasoning.\n\nAnswer: B")
    assert (s["acc"], s["miss"], s["acc_anchored"]) == (1.0, 0.0, 1.0)
    s = enc._score(_doc("B"), "Reasoning.\n\nAnswer: B or C")          # the official reader credits the first letter, the strict reader does not
    assert (s["acc"], s["miss"], s["acc_anchored"]) == (1.0, 0.0, 0.0)
    s = enc._score(_doc("A"), "Reasoning.\n\nAnswer: B")
    assert (s["acc"], s["miss"], s["acc_anchored"]) == (0.0, 0.0, 0.0)      # a wrong letter is wrong, not a miss
    s = enc._score(_doc("A"), "answer: a")                              # lower-case: extracted ('a'), wrong (the comparison is case-sensitive), no miss
    assert (s["acc"], s["miss"]) == (0.0, 0.0)
    s = enc._score(_doc("A"), "Nothing to see here, sorry.")
    assert (s["acc"], s["miss"], s["acc_anchored"]) == (0.0, 1.0, 0.0)
    for empty in ("", "   \n"):                                         # an empty text (an unclosed think block under --strict-reasoning) is a miss and wrong
        s = enc._score(_doc("A"), empty)
        assert (s["acc"], s["miss"], s["acc_anchored"]) == (0.0, 1.0, 0.0)
    s = enc._score(_doc("A", options=["i.ii.", "iii.iv.", "v.vi.", "vii.viii.", "ix.x."]), "Reasoning.\n\niii.iv.")      # option-text fallback, mapped to its letter
    assert s["acc"] == 0.0 and s["miss"] == 0.0
    s = enc._score(_doc("B", options=["i.ii.", "iii.iv.", "v.vi.", "vii.viii.", "ix.x."]), "Reasoning.\n\niii.iv.")
    assert s["acc"] == 1.0 and s["miss"] == 0.0


def test_metric_breakdown_by_discipline_and_corpus_values():
    metric = enc.encyclo_k_chat_metric(1)
    assert list(metric.metric_name) == list(enc.NAMES) and len(enc.NAMES) == 3 + 11
    assert metric.higher_is_better["miss"] is False and metric.higher_is_better["acc"] is True
    fn = metric.sample_level_fn
    a = fn.compute(_doc("B", discipline="Law"), _resp("Answer: B"))
    assert a["acc"] == 1.0 and a["acc_law"] == 1.0 and math.isnan(a["acc_science"]) and "pass@2" not in a
    b = fn.compute(_doc("B", discipline="Science"), _resp("Answer: C"))
    assert b["acc"] == 0.0 and b["acc_science"] == 0.0 and math.isnan(b["acc_law"])
    c = fn.compute(_doc("B", discipline="Science"), _resp("Answer: B"))
    out = {k: metric.corpus_level_fn[k]([x[k] for x in (a, b, c)]) for k in enc.NAMES}
    assert out["acc"] == 2 / 3 and out["acc_law"] == 1.0 and out["acc_science"] == 0.5 and out["miss"] == 0.0
    assert math.isnan(out["acc_history"])                                  # a discipline without a member in this slice
    assert sorted(enc.DISCIPLINES) == enc.DISCIPLINES and len(enc.DISCIPLINE_KEYS) == 11


def test_the_scored_text_is_the_final_text_after_the_think_block():
    fn = enc.encyclo_k_chat_metric(1).sample_level_fn
    # the harness cuts at the LAST </think> and gives "" for an unclosed block (--strict-reasoning); the metric reads final_text only
    closed = ModelResponse(text=["<think>Answer: A</think>\n\nAnswer: B"], text_post_processed=["\n\nAnswer: B"])
    assert fn.compute(_doc("B"), closed)["acc"] == 1.0
    assert fn.compute(_doc("A"), closed)["acc"] == 0.0                      # the answer inside the reasoning does not count
    unclosed = ModelResponse(text=["<think>Answer: B"], text_post_processed=[""])
    s = fn.compute(_doc("B"), unclosed)
    assert s["acc"] == 0.0 and s["miss"] == 1.0
    two = fn.compute(_doc("B"), ModelResponse(text=["Answer: B", "Answer: C"], text_post_processed=["Answer: B", "Answer: C"]))
    assert two["acc"] == 0.5 and two["miss"] == 0.0                         # several samples: the mean over the samples


def test_a_pathological_answer_cannot_hang_the_scorer():
    """The official patterns backtrack exponentially on "Answer: " + a run of "*" ($ { ( [ \\); the official 5 s timeout cannot interrupt `re` (a signal waits for the C call),
    `regex` can (the official TODO). With a 0.3 s timeout the search ends with no match."""
    if not enc._TIMEOUT:
        return
    old = enc.REGEX_TIMEOUT
    enc.REGEX_TIMEOUT = 0.3
    try:
        t0 = time.time()
        assert enc.official_answer("Answer: " + "*" * 60 + "x", OPTS) is None
        assert time.time() - t0 < 10
    finally:
        enc.REGEX_TIMEOUT = old


def test_prompt_is_the_official_prompt_character_for_character():
    q = "Review the following statements and identify those that are accurate: \ni.First {braces} statement.\nii.Second statement.\n"
    options = ["i.", "ii.", "i.ii.", "none."]
    want = (
        "Answer the following multiple choice question. Please carefully analyze each statement and option, and end your response with 'Answer: $OPTION_NUMBER' (no quotes), "
        "where OPTION_NUMBER is A, B, C, D.\n\n" + q + "\nA) i.\nB) ii.\nC) i.ii.\nD) none.\n"
    )
    assert enc.official_prompt(q, options) == want
    line = {"question": q, "options": options, "answer_letter": "C", "discipline": "Law", "field": "f", "subfield": "s", "difficulty": "middle", "uuid": "u-1"}
    doc = enc.encyclo_k_chat_prompt(line, "encyclo_k_chat")
    assert doc.query == want and doc.query.endswith("D) none.\n") and doc.instruction is None
    assert doc.choices == ["A", "B", "C", "D"] and doc.gold_index == 2
    assert doc.specific["options"] == options and doc.specific["discipline"] == "Law" and doc.specific["lang"] == "en"
    try:
        enc.encyclo_k_chat_prompt({**line, "answer_letter": "E"}, "t")           # no silent default for a gold outside the options
    except ValueError:
        pass
    else:
        raise AssertionError("a gold letter outside the options must raise")
    assert enc.encyclo_k_chat_prompt({**line, "question": "统计学" * 10 + " statement"}, "t").specific["lang"] == "zh"


def test_task_config():
    (task,) = enc.TASKS_TABLE
    assert task.name == "encyclo_k_chat" and task.hf_repo == "m-a-p/Encyclo-K" and task.hf_revision == REVISION == "0f81da9652b9c4d7bad73ce9576d81901d9441f2"
    assert task.hf_subset == "default" and tuple(task.evaluation_splits) == ("test",) and task.few_shots_split is None
    assert task.generation_size == 32768 and task.stop_sequence == [] and task.version == 0
    assert task.metrics[0].sample_level_fn.n == 1                              # the official run is one greedy sample per question
    assert task.name.endswith("_chat")


def test_the_full_dataset_matches_the_paper_and_the_official_prompt():
    """5,038 questions, the paper's per-discipline counts (Table 2), 4-8 options, a gold letter inside the options, the dataset's own `prompt` column == the official loader's text."""
    import collections
    import json

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return
    path = try_to_load_from_cache("m-a-p/Encyclo-K", "encyclo-k_all_prompt.jsonl", revision=REVISION, repo_type="dataset")
    if not isinstance(path, str):
        return                                                                 # the data is not in the Hugging Face cache of this machine
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(rows) == 5038
    docs = [enc.encyclo_k_chat_prompt(r, "encyclo_k_chat") for r in rows]
    assert all(d.query == r["prompt"] for d, r in zip(docs, rows)), "the official prompt rebuilt from question + options is the dataset's `prompt` column"
    assert collections.Counter(r["discipline"] for r in rows) == {
        "Science": 1242, "Engineering": 881, "Literature": 825, "Medicine": 736, "History": 686, "Law": 189, "Economics": 124, "Agriculture": 112, "Education": 109, "Management": 73,
        "Philosophy": 61}
    assert set(r["discipline"] for r in rows) == set(enc.DISCIPLINES)
    assert {len(r["options"]) for r in rows} == {4, 5, 6, 7, 8} and all(d.gold_index < len(d.choices) for d in docs)
    assert all(r["options"][ord(r["answer_letter"]) - 65] == r["answer"] for r in rows)       # the gold letter points at the answer text
    assert collections.Counter(d.specific["lang"] for d in docs) == {"en": 3286, "zh": 1752}   # the paper's Table 2 split
    assert all(r["options"] and all(o for o in r["options"]) for r in rows) and len({r["uuid"] for r in rows}) == 5038


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
