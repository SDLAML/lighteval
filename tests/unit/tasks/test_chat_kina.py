"""Tests of kina_chat (src/lighteval/tasks/tasks/chat_kina.py): the official prompt (github.com/2077AI/KINA @ 69a97f30, src/kina_bench/utils.py `prompt_template` / `get_messages`), the
official answer reader (lighteval 0.9.2's `gpqa_instruct_pass_at_1_*n` metric with `IndicesExtractionConfig("NativeLetters")`, try_extract_without_anchor=True), the metric and its
per-discipline breakdown.

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_kina.py`. No GPU; the dataset test skips when the pinned data is not in the Hugging Face cache.
The expected letters of `test_official_reader_cases` were derived by reading the 0.9.2 source (priority order, rightmost match); the 0.9.2 package itself was NOT executed here
(the review page says so), the seven pattern strings of `test_patterns_are_the_official_ones` are written out by hand from that source and compared with the module's.
"""

import json
import math
import re

from lighteval.metrics.chat_metrics import anchored_letter
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_kina as kina
from lighteval.tasks.tasks.kina import _COT_INSTRUCTION
from lighteval.tasks.tasks.kina_base import REVISION


# utils.py l.110-114, verbatim (a plain, non-f string: the placeholders are `str.format` fields)
OFFICIAL_TEMPLATE = """Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J. Think step by step before answering.
Question:
{question}
Options:
{options}"""

LETTERS = "ABCDEFGHIJ"


def _doc(gold="B", discipline="Law"):
    return Doc(task_name="kina_chat", query="q", choices=list(LETTERS), gold_index=LETTERS.index(gold), specific={"index": 0, "discipline": discipline, "path": discipline})


def _resp(*texts):
    return ModelResponse(text=list(texts), text_post_processed=list(texts))


def _official_messages(item: dict) -> str:
    """utils.py `load_data` + `get_messages` for one raw item, written independently of the task code (the key map, `.strip().upper()` of the keys, `.strip()` of the answers, the
    question unchanged, sorted keys)."""
    options = {}
    for opt in item["options"]:
        key = opt.get("key")
        if not key or not isinstance(key, str):
            continue
        key = key.strip().upper()
        options[key] = {"answer": (opt.get("answer") or "").strip() if opt.get("answer") is not None else ""}
    keys = sorted(options.keys())
    option_str = "\n".join([f"{letter}: {options[letter]['answer'].strip()}" for letter in keys])
    return OFFICIAL_TEMPLATE.format(question=item["question"], options=option_str)


# The seven patterns of lighteval 0.9.2 `lazy_indices_regex` (English, A-J, try_extract_without_anchor=True) with the f-strings evaluated by hand:
#   indice = (?P<indices>A|B|C|D|E|F|G|H|I|J)   full stop [\.\.]   comma [,\,]   colon [:\:]   space \<space>
#   answer_prefix = (^|\ )(?:\*\*)?   answer_suffix = (?:\*\*)?(?:[\.\.]|[,\,]|[:\:]|\ |$)   answer_re = prefix + indice + suffix   answer_re_start = ^(?:\*\*)? + indice + suffix
INDICE = r"(?P<indices>A|B|C|D|E|F|G|H|I|J)"
SUFFIX = r"(?:\*\*)?(?:[\.\.]|[,\,]|[:\:]|\ |$)"
ANSWER_RE = r"(^|\ )(?:\*\*)?" + INDICE + SUFFIX
OFFICIAL_PATTERNS = (
    (r"(?i:final answer is)\:?\s*" + INDICE + r"\.?\s?I hope", 0),
    (r"(?i:final answer.{0,100}?)\s+is\:?\s*" + INDICE, 50),
    (r"(?i:answer)[:\:].{0,50}?" + ANSWER_RE, 100),
    (r"(?i:answer).{0,50}?" + ANSWER_RE, 150),
    (r"^(?:\*\*)?" + INDICE + SUFFIX, 200),
    (ANSWER_RE, 250),
    (INDICE, 300),
)


def test_patterns_are_the_official_ones():
    assert tuple(kina.OFFICIAL_PATTERNS) == OFFICIAL_PATTERNS
    assert [p for _, p in kina.OFFICIAL_PATTERNS] == sorted(p for _, p in kina.OFFICIAL_PATTERNS)       # tried in ascending priority
    assert all(pat.flags & re.IGNORECASE == 0 and pat.flags & re.DOTALL == 0 and pat.flags & re.MULTILINE == 0 for pat, _ in kina._COMPILED)      # compiled without flags, as the official
    assert kina._LETTERS == "ABCDEFGHIJ"                    # `len(choices)` = 10 in the official `judge_score`


# (name, the generated final text, the letter the official reader takes). Expected values: priority order of the 0.9.2 source, the match that ends last wins inside a priority.
CASES = [
    ("plain anchor", "The enzyme is inhibited by the product.\n\nAnswer: C", "C"),
    ("bold whole line", "Reasoning here.\n\n**Answer: C**", "C"),
    ("bold label and bold letter", "**Answer:** **J**", "J"),
    ("letter J", "Answer: J", "J"),
    ("letter I", "Answer: I", "I"),
    ("letter A", "Answer: A", "A"),
    ("letter with full stop", "Answer: F.", "F"),
    ("letter then explanation", "Answer: C Because the lease was void.", "C"),
    ("letter then a trailing newline (`$` also matches before a final newline)", "Answer: C\n", "C"),
    ("explanation after the answer line: the newline after the letter is no suffix, so no anchor matches and the last capital A-J of the text wins (official quirk)",
     "Answer: C\n\nExplanation: DNA is not involved, option D is a distractor.", "D"),
    ("two answer lines: the last counts", "Answer: B\nOn reflection, that is wrong.\nAnswer: E", "E"),
    ("hedge: the first letter", "Answer: B or C", "B"),
    ("final answer with I hope", "Therefore, the final answer is C. I hope it is correct", "C"),
    ("no anchor, sentence", "The correct option is D", "D"),
    ("boxed letter only", "$\\boxed{C}$", "C"),
    ("letter on the line after the word", "Answer:\nC", "C"),
    ("parenthesised letter", "Answer: (C)", "C"),
    ("option text after a bracket: a capital in the text wins", "Answer: B) Aspirin", "A"),
    ("pronoun I at the start", "I cannot determine this.", "I"),
    ("lower case answer, no capital A-J at all", "answer: c", None),
    ("only digits and lower case", "no capitals here 123", None),
    ("empty", "", None),
]


def test_official_reader_cases():
    for name, text, want in CASES:
        assert kina.official_letter(text) == want, (name, text, kina.official_letter(text))


def test_every_letter_in_every_answer_form():
    for letter in LETTERS:        # J is the last of the ten, A and I are also an article and a pronoun
        for form in ("Answer: {}", "**Answer: {}**", "Answer: {}.", "Answer: **{}**", "The last line.\n\nAnswer: {}\n", "Some text. Answer: {}"):
            assert kina.official_letter(form.format(letter)) == letter, (form, letter)
        assert kina.official_letter(f"Answer: {letter}") == letter
        assert kina.official_letter(f"Answer: {letter.lower()}") == "A"        # a lower-case letter is no letter: the capital A of the word "Answer" is the last capital A-J (the official quirk)


def test_the_strict_and_the_official_reader_differ_where_documented():
    s = kina._score(_doc("D"), "The correct option is D")
    assert s["acc"] == 1.0 and s["acc_anchored"] == 0.0 and s["miss"] == 0.0                       # the official reader credits an unanchored letter, the strict one does not
    s = kina._score(_doc("B"), "Answer: B or C")
    assert s["acc"] == 1.0 and s["acc_anchored"] == 0.0                                            # a hedge: the first letter officially, no letter in the strict reader
    s = kina._score(_doc("B"), "Answer: B")
    assert (s["acc"], s["acc_anchored"], s["miss"]) == (1.0, 1.0, 0.0)
    assert anchored_letter("Answer: J", 10) == "J" and anchored_letter("Answer: J", 9) is None       # the strict reader checks the option range


def test_empty_and_letterless_answers_are_a_miss_and_wrong():
    for empty in ("", "   \n"):
        s = kina._score(_doc("A"), empty)
        assert (s["acc"], s["miss"], s["acc_anchored"]) == (0.0, 1.0, 0.0)
    s = kina._score(_doc("A"), "i do not know")           # no capital A-J: the official `max([])` has no value, here a miss
    assert (s["acc"], s["miss"]) == (0.0, 1.0)
    s = kina._score(_doc("A"), "Answer: C")
    assert (s["acc"], s["miss"]) == (0.0, 0.0)             # a wrong letter is wrong, not a miss


def test_metric_breakdown_by_discipline_and_corpus_values():
    metric = kina.kina_chat_metric()
    assert list(metric.metric_name) == list(kina.NAMES) and len(kina.NAMES) == 3 + 12 and metric.sample_level_fn.n == 4 == kina.N_SAMPLES
    assert metric.higher_is_better["miss"] is False and metric.higher_is_better["acc"] is True
    assert list(kina.DISCIPLINE_KEYS.values())[:3] == ["acc_agronomy", "acc_economics", "acc_education"] and "acc_literature_and_arts" in kina.DISCIPLINE_KEYS.values()
    fn = metric.sample_level_fn
    a = fn.compute(_doc("B", discipline="Law"), _resp("Answer: B", "Answer: B", "Answer: C", "Answer: B"))      # avg@4: 3 of 4 samples right
    assert a["acc"] == 0.75 and a["acc_law"] == 0.75 and math.isnan(a["acc_science"]) and "pass@2" not in a
    b = fn.compute(_doc("B", discipline="Science"), _resp(*["Answer: C"] * 4))
    c = fn.compute(_doc("B", discipline="Science"), _resp(*["Answer: B"] * 4))
    out = {k: metric.corpus_level_fn[k]([x[k] for x in (a, b, c)]) for k in kina.NAMES}
    assert math.isclose(out["acc"], (0.75 + 0 + 1) / 3) and out["acc_law"] == 0.75 and out["acc_science"] == 0.5 and out["miss"] == 0.0
    assert math.isnan(out["acc_history"])                                                         # a discipline without a member in this slice


def test_the_scored_text_is_the_final_text_after_the_think_block():
    fn = kina.kina_chat_metric().sample_level_fn
    # the harness cuts at the LAST </think> and gives "" for an unclosed block (--strict-reasoning); the metric reads final_text only
    closed = ModelResponse(text=["<think>Answer: A</think>\n\nAnswer: B"], text_post_processed=["\n\nAnswer: B"])
    assert fn.compute(_doc("B"), closed)["acc"] == 1.0
    assert fn.compute(_doc("A"), closed)["acc"] == 0.0                                            # the answer inside the reasoning does not count
    unclosed = ModelResponse(text=["<think>I think it is B. Answer: B"], text_post_processed=[""])
    s = fn.compute(_doc("B"), unclosed)
    assert s["acc"] == 0.0 and s["miss"] == 1.0                                                   # official code: content None -> the REASONING is scored; here an unclosed block is 0
    nothink = ModelResponse(text=["Step 1 ... Step 2 ...\n\n**Answer: J**"], text_post_processed=["Step 1 ... Step 2 ...\n\n**Answer: J**"])
    assert fn.compute(_doc("J"), nothink)["acc"] == 1.0 and fn.compute(_doc("I"), nothink)["acc"] == 0.0
    mixed = ModelResponse(text=["Answer: B", "Answer: C", "", "Answer: B"], text_post_processed=["Answer: B", "Answer: C", "", "Answer: B"])
    s = fn.compute(_doc("B"), mixed)
    assert s["acc"] == 0.5 and s["miss"] == 0.25                                                   # a sample cut at the budget counts as wrong and as a miss


def test_prompt_is_the_official_prompt_character_for_character():
    assert _COT_INSTRUCTION == OFFICIAL_TEMPLATE.split("\nQuestion:")[0], "kina.py's _COT_INSTRUCTION is the official instruction line"
    assert kina.PROMPT_TEMPLATE == OFFICIAL_TEMPLATE
    opts = [{"key": k, "answer": f"  option {k} text \n", "explanation": f"never shown {k}"} for k in LETTERS]
    item = {"index": 7, "discipline": "Law/Civil Law/Contracts", "question": "  Which?\r\n1. first\r\n2. second\n", "options": opts, "correct_answer": " c "}
    doc = kina.kina_chat_prompt(item, "kina_chat")
    want = _official_messages(item)
    assert doc.query == want and doc.instruction is None
    assert "never shown" not in doc.query and doc.query.endswith("J: option J text") and "\nQuestion:\n  Which?\r\n1. first\r\n2. second\n\nOptions:\nA: option A text\nB: option B text\n" in doc.query
    assert doc.choices == list(LETTERS) and doc.gold_index == 2 and doc.specific == {"index": 7, "discipline": "Law", "path": "Law/Civil Law/Contracts"}
    shuffled = {**item, "options": list(reversed(opts))}
    assert kina.kina_chat_prompt(shuffled, "t").query == want, "options are ordered by their key, not by list position"
    for bad in ("K", "", "AB"):
        try:
            kina.kina_chat_prompt({**item, "correct_answer": bad}, "t")
        except ValueError:
            pass
        else:
            raise AssertionError("a gold letter outside A-J must raise")


def test_task_config():
    (task,) = kina.TASKS_TABLE
    assert task.name == "kina_chat" and task.hf_repo == "2077AIDataFoundation/KINA" and task.hf_revision == REVISION == "860c64e856375ed42929737fb8b8f37272c97065"
    assert task.hf_subset == "default" and tuple(task.evaluation_splits) == ("test",) and task.few_shots_split is None
    assert task.generation_size == 32768 and task.stop_sequence == [] and task.version == 0       # the paper's "32K maximum new tokens"
    assert task.metrics[0].sample_level_fn.n == 4                                                   # avg@4
    assert task.name.endswith("_chat") and kina.OFFICIAL_COMMIT == "69a97f305af11d0001854d3392b8410f92fbaae3"


def test_the_full_dataset_matches_the_paper_and_the_official_prompt():
    """899 questions, the paper's per-discipline counts (Figure 2: Engineering 34.26%, Science 22.36% ...), ten options, the gold inside A-J, every prompt equal to the independent
    port of the official `get_messages`."""
    import collections

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return
    path = try_to_load_from_cache("2077AIDataFoundation/KINA", "KINA.json", revision=REVISION, repo_type="dataset")
    if not isinstance(path, str):
        return                                                                                     # the data is not in the Hugging Face cache of this machine
    rows = json.load(open(path, encoding="utf-8"))
    assert len(rows) == 899 and [r["index"] for r in rows] == list(range(899))
    docs = [kina.kina_chat_prompt(r, "kina_chat") for r in rows]
    assert all(d.query == _official_messages(r) for d, r in zip(docs, rows)), "the task prompt equals the official get_messages on every item"
    assert len({d.query for d in docs}) == 899
    assert collections.Counter(d.specific["discipline"] for d in docs) == {
        "Engineering": 308, "Science": 201, "Medicine": 81, "Law": 65, "Literature and Arts": 65, "Agronomy": 40, "Economics": 34, "Education": 31, "Management": 29, "Sociology": 19,
        "History": 13, "Philosophy": 13}
    assert set(kina.DISCIPLINES) == set(d.specific["discipline"] for d in docs) and len(kina.DISCIPLINES) == 12
    assert collections.Counter(r["correct_answer"] for r in rows) == {"A": 187, "J": 96, "G": 96, "C": 90, "B": 87, "D": 87, "E": 75, "F": 69, "H": 66, "I": 46}
    assert all([o["key"] for o in r["options"]] == list(LETTERS) and all(o["answer"].strip() for o in r["options"]) for r in rows)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
