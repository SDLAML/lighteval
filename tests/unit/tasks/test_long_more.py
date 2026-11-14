"""Tests of the long-context tasks of the fork module tasks/tasks/long_more.py: LongBench v2 base, GSM-Infinite (base + chat), MRCR (base + chat), FRAMES (chat), and the
`chat_messages` hook of the prompt manager.

Plain functions: run with pytest, or `python tests/unit/tasks/test_long_more.py`. Nothing here needs a GPU. Tests that read the prepared data
(titan_eval/evals/long_data.py, <cache>/xdg/...) or a tokenizer skip when it is not on disk.
"""

import json
import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import long_more as lm


QWEN = "/e/data1/trustllmd/wang55/hf_models/Qwen3.5-2B-Base"
XDG = Path(os.environ.get("TITAN_EVAL_CACHE", "/e/scratch/trustllm-eu/wang55/titan_eval/.cache")) / "xdg"


class FakeTok:
    """Word-level tokenizer: one token per whitespace word (as test_chat_long)."""

    def encode(self, text, add_special_tokens=False):
        return text.split(" ")

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def env(ctx=None, new=None, tok="fake"):
    d = {}
    if ctx is not None:
        d["TITAN_EVAL_MAX_MODEL_LEN"] = str(ctx)
    if new is not None:
        d["TITAN_EVAL_MAX_NEW_TOKENS"] = str(new)
    if tok:
        d["TITAN_EVAL_TOKENIZER"] = tok
    return mock.patch.dict(os.environ, d, clear=False)


def no_env():
    return mock.patch.dict(os.environ, {k: "" for k in ("TITAN_EVAL_MAX_MODEL_LEN", "TITAN_EVAL_MAX_NEW_TOKENS", "TITAN_EVAL_TOKENIZER")}, clear=False)


def score(metric, doc, text):
    return metric.sample_level_fn.compute(doc, ModelResponse(text=[text]))


def lb2_line(context="word " * 50, difficulty="hard", length="long", answer="C", question="What is the answer?"):
    return {"_id": "t1", "domain": "Legal", "sub_domain": "Case law", "difficulty": difficulty, "length": length, "question": question,
            "choice_A": "alpha", "choice_B": "beta", "choice_C": "gamma", "choice_D": "delta", "answer": answer, "context": context}


# ---------------------------------------------------------------------------------------------------------- LongBench v2 base
class LongBenchBase(unittest.TestCase):
    def test_prompt_is_official_template_plus_answer_prefix(self):
        with no_env():
            d = lm.lb2_base_prompt_fn(8192)(lb2_line(context="  ctx  ", question=" Q? "), "t")
        self.assertIn("<text>\nctx\n</text>", d.query)
        self.assertIn("What is the correct answer to this question: Q?\nChoices:\n(A) alpha\n(B) beta\n(C) gamma\n(D) delta\n", d.query)
        self.assertTrue(d.query.endswith('Format your response as follows: "The correct answer is (insert answer here)".\nThe correct answer is ('))
        self.assertEqual((d.choices, d.gold_index), (["A", "B", "C", "D"], 2))
        self.assertEqual(d.specific["n_tokens"], -1)               # no tokenizer: nothing counted, nothing cut
        self.assertFalse(d.specific["truncated"])

    def test_no_chat_template_fields(self):
        with no_env():
            d = lm.lb2_base_prompt_fn(8192)(lb2_line(), "t")
        self.assertIsNone(d.instruction)
        self.assertNotIn("<|im_start|>", d.query)

    def test_middle_cut_to_context_minus_generation_minus_margin_minus_prefix(self):
        ctx = 400
        with env(ctx, None), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            d = lm.lb2_base_prompt_fn(ctx)(lb2_line(context=" ".join(f"w{i}" for i in range(1000))), "t")
        tail = "\n" + lm.LB2B_PREFIX
        budget = ctx - lm.LB2B_NEW - lm.LB2B_MARGIN - len(FakeTok().encode(tail))
        self.assertTrue(d.query.endswith(tail))
        self.assertEqual(len(FakeTok().encode(d.query[: -len(tail)])), budget)       # exactly the budget, then the tail
        self.assertTrue(d.specific["truncated"])
        self.assertGreater(d.specific["n_tokens"], ctx)
        self.assertIn("(D) delta", d.query)                          # the choices and the instruction at the end survive
        self.assertNotIn("w500 ", d.query)                           # the middle does not

    def test_short_prompt_is_not_cut(self):
        with env(8192, None), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            d = lm.lb2_base_prompt_fn(8192)(lb2_line(context="a b c"), "t")
        self.assertFalse(d.specific["truncated"])
        self.assertGreater(d.specific["n_tokens"], 0)
        self.assertIn("a b c", d.query)

    def test_run_context_wins_over_the_nominal_length(self):
        with env(1000, None), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            d = lm.lb2_base_prompt_fn(8192)(lb2_line(context=" ".join(["x"] * 900)), "t")
        self.assertFalse(d.specific["truncated"])                  # 900 words + template fit 1000 - 8 - 32 - tail

    def test_no_room_is_an_error(self):
        with env(16, None), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            with self.assertRaises(ValueError):
                lm.lb2_base_prompt_fn(8192)(lb2_line(), "t")

    def test_letter_extraction(self):
        for text, want in [("B)", "B"), ("B", "B"), (" C.", "C"), ("(D)", "D"), ("A) alpha", "A"), ("a)", None), ("Answer", None), ("E)", None), ("", None), ("BC", None),
                           ("\nA", "A"), ("1", None)]:
            self.assertEqual(lm.lb2_base_letter(text), want, text)

    def _doc(self, **spec):
        base = {"difficulty": "hard", "length": "long", "n_tokens": 70000, "truncated": True}
        base.update(spec)
        return Doc(task_name="t", query="q", choices=list("ABCD"), gold_index=2, specific=base)

    def test_scores_and_buckets(self):
        m = lm.longbench_v2_base_metric()
        ok = score(m, self._doc(), "C)")
        self.assertEqual(ok["acc"], 1.0)
        self.assertEqual((ok["acc_hard"], ok["acc_long"], ok["acc_truncated"], ok["acc_tok_64k_128k"], ok["truncated"], ok["no_letter"]), (1.0, 1.0, 1.0, 1.0, 1.0, 0.0))
        for name in ("acc_easy", "acc_short", "acc_medium", "acc_untruncated", "acc_tok_0_16k", "acc_tok_16k_32k", "acc_tok_32k_64k", "acc_tok_128k_up"):
            self.assertTrue(math.isnan(ok[name]), name)
        bad = score(m, self._doc(n_tokens=9000, truncated=False, difficulty="easy", length="short"), "B)")
        self.assertEqual((bad["acc"], bad["acc_easy"], bad["acc_short"], bad["acc_untruncated"], bad["acc_tok_0_16k"]), (0.0, 0.0, 0.0, 0.0, 0.0))
        none = score(m, self._doc(), "I think")
        self.assertEqual((none["acc"], none["no_letter"]), (0.0, 1.0))
        self.assertEqual(score(m, self._doc(), "")["acc"], 0.0)

    def test_token_bucket_edges(self):
        m = lm.longbench_v2_base_metric()
        for n, name in [(16384, "acc_tok_0_16k"), (16385, "acc_tok_16k_32k"), (32768, "acc_tok_16k_32k"), (65536, "acc_tok_32k_64k"), (131072, "acc_tok_64k_128k"), (131073, "acc_tok_128k_up")]:
            s = score(m, self._doc(n_tokens=n), "C")
            self.assertEqual(s[name], 1.0, (n, name))
            self.assertEqual(sum(1 for k in lm.LB2B_TOKEN_BUCKETS if not math.isnan(s[k])), 1, n)

    def test_no_tokenizer_leaves_token_metrics_nan(self):
        m = lm.longbench_v2_base_metric()
        s = score(m, self._doc(n_tokens=-1, truncated=False), "C")
        self.assertEqual(s["acc"], 1.0)
        for name in (*lm.LB2B_TOKEN_BUCKETS, "acc_untruncated", "acc_truncated", "truncated"):
            self.assertTrue(math.isnan(s[name]), name)

    def test_aggregation_is_mean_over_members(self):
        m = lm.longbench_v2_base_metric()
        fns = m.corpus_level_fn
        self.assertAlmostEqual(fns["acc_hard"]([1.0, float("nan"), 0.0, float("nan"), 1.0]), 2 / 3)
        self.assertAlmostEqual(fns["acc"]([1.0, 0.0, 1.0, 0.0]), 0.5)
        self.assertTrue(math.isnan(fns["acc_easy"]([float("nan")] * 3)))

    def test_five_tasks_with_their_own_length(self):
        names = [c.name for c in lm.TASKS_TABLE if c.name.startswith("longbench_v2_base")]
        self.assertEqual(names, [f"longbench_v2_base_{n}" for n in (8192, 16384, 32768, 65536, 131072)])
        cfg = [c for c in lm.TASKS_TABLE if c.name == "longbench_v2_base_8192"][0]
        self.assertEqual((cfg.hf_repo, cfg.hf_revision, cfg.generation_size, cfg.stop_sequence), ("zai-org/LongBench-v2", "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9", 8, ["\n"]))


# --------------------------------------------------------------------------------------------------------------- GSM-Infinite
class GsmInfiniteParser(unittest.TestCase):
    """The official parser (eval_realistic.py criteriaoutput): a copy of its loop, applied to one text, is the reference of every case below."""

    @staticmethod
    def official(text):
        import re

        def is_integer(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        t = re.sub(".\x08", "b", text).lower()
        keywords = ["answer: ", "solution: ", "oxed{", "**answer:** ", "**answer: ", "final answer: answer: ", "\nanswer: ", r"\text{answer: } ", "is ", "answer: "]
        keywordsend = [".", ".", "}", ".", "**", ".", ".", None, ".", "\n"]
        b = c = -1
        cnt = 0
        while not (b != -1 and c != -1) and cnt < len(keywords):
            if keywords[cnt] in ["oxed{", "is "]:
                b = t.rfind(keywords[cnt])
            else:
                b = t.find(keywords[cnt])
            if b != -1:
                if keywordsend[cnt] is None:
                    c = b + len(keywords[cnt])
                    while t[c].isdigit():
                        c += 1
                else:
                    c = t.find(keywordsend[cnt], b + len(keywords[cnt]))
                if c == -1:
                    c = len(t)
            cnt += 1
            if not is_integer(t[b + len(keywords[cnt - 1]):c]):
                b = c = -1
                continue
        if b == -1:
            return None
        try:
            return int(t[b + len(keywords[cnt - 1]):c])
        except ValueError:
            return None

    CASES = ["Define A as 3; so A = 3. Answer: 6.", "Answer: 12", "so the answer is 7.", "\\boxed{42}", "**Answer:** 5.", "The final answer: Answer: 9.", "no number here", "",
             "Answer: -3.", "Answer: six.", "Answer: 4.\n\nProblem: ... Answer: 99.", "Define x as 2. x = 2\nAnswer: 8\nExplanation", "This is 5.", "answer: 7 and then solution: 8.",
             "The ANSWER: 15.", "Solution: 10.", "answer: 3.5.", "bla Answer:    7."]

    def test_same_as_the_official_loop(self):
        for text in self.CASES:
            self.assertEqual(lm.gsm_extract(text), self.official(text), text)

    def test_known_values(self):
        self.assertEqual(lm.gsm_extract("Define A as 3; so A = 3. Answer: 6."), 6)
        self.assertEqual(lm.gsm_extract("Answer: -3."), -3)
        self.assertIsNone(lm.gsm_extract("I do not know"))
        self.assertEqual(lm.gsm_extract("Answer: 4.\n\nProblem: ... Answer: 99."), 4)       # the FIRST "answer: " (official)
        self.assertEqual(lm.gsm_extract("so the result is 7"), 7)                          # "is " keyword, last occurrence

    def test_digits_to_the_end_do_not_raise(self):
        self.assertEqual(lm.gsm_extract("see \\text{answer: } 12"), 12)      # the official loop raises IndexError at the end of the text; the guard returns the digits


def gsm_line(op=5, mode="normalforward", answer=6, problem="P" * 30, question="Q?"):
    msgs = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Instruction. Answer: xxx."},
            {"role": "user", "content": f"Problem: {problem} Question: {question} Solution: "}]
    return {"id": f"hard_8k_op{op}_000", "src_id": 1, "op": op, "template": "crazy_zootopia", "mode": mode, "length": "8k", "d": 3, "answer": answer, "solution": "x", "messages": json.dumps(msgs),
            "n_chars": 100}


class GsmInfinite(unittest.TestCase):
    def test_chat_doc_carries_the_messages_as_given(self):
        with no_env():
            d = lm.gsm_doc(gsm_line(), "t", "hard", chat=True)
        msgs = d.specific["chat_messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "user"])
        self.assertTrue(msgs[2]["content"].endswith("Solution: "))     # verbatim, trailing space included
        self.assertEqual((d.choices, d.gold_index), (["6"], 0))
        self.assertEqual((d.specific["op"], d.specific["mode"]), (5, "normalforward"))

    def test_base_doc_is_plain_text_without_system_and_trailing_space(self):
        with no_env():
            d = lm.gsm_doc(gsm_line(), "t", "hard", chat=False)
        self.assertEqual(d.query, "Instruction. Answer: xxx.\n\nProblem: " + "P" * 30 + " Question: Q? Solution:")
        self.assertNotIn("chat_messages", d.specific)
        self.assertNotIn("helpful assistant", d.query)

    def test_item_that_does_not_fit_is_skipped_with_a_tiny_prompt(self):
        long_line = gsm_line(problem=" ".join(["w"] * 5000))
        with env(4000, 1000), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            d = lm.gsm_doc(long_line, "t", "hard", chat=True)
            d2 = lm.gsm_doc(gsm_line(), "t", "hard", chat=True)
        self.assertTrue(d.specific["skipped"])
        self.assertEqual(d.specific["chat_messages"], [{"role": "user", "content": lm.SKIP_USER}])
        self.assertFalse(d2.specific["skipped"])
        self.assertEqual(d2.specific["chat_messages"][0]["role"], "system")

    def test_no_env_never_skips(self):
        with no_env():
            d = lm.gsm_doc(gsm_line(problem=" ".join(["w"] * 50000)), "t", "hard", chat=False)
        self.assertFalse(d.specific["skipped"])

    def _score(self, text, **kw):
        with no_env():
            d = lm.gsm_doc(gsm_line(**kw), "t", "hard", chat=False)
        return score(lm.gsm_metric(), d, text)

    def test_correct_wrong_and_missing(self):
        s = self._score("Define A as 3. Answer: 6.", op=5)
        self.assertEqual((s["acc"], s["acc_op5"], s["acc_forward"], s["no_answer"], s["skipped"]), (1.0, 1.0, 1.0, 0.0, 0.0))
        self.assertTrue(math.isnan(s["acc_op4"]) and math.isnan(s["acc_reverse"]))
        w = self._score("Answer: 7.", op=5, mode="forwardreverse")
        self.assertEqual((w["acc"], w["acc_op5"], w["acc_reverse"], w["no_answer"]), (0.0, 0.0, 0.0, 0.0))
        n = self._score("I cannot say", op=5)
        self.assertEqual((n["acc"], n["no_answer"]), (0.0, 1.0))
        self.assertEqual(self._score("", op=5)["no_answer"], 1.0)

    def test_skipped_items_are_nan_everywhere_but_the_skip_count(self):
        d = Doc(task_name="t", query="q", choices=["6"], gold_index=0, specific={"op": 5, "mode": "normalforward", "skipped": True})
        s = score(lm.gsm_metric(), d, "OK")
        self.assertEqual(s["skipped"], 1.0)
        self.assertTrue(all(math.isnan(v) for k, v in s.items() if k != "skipped"))

    def test_corpus_aggregation(self):
        fns = lm.gsm_metric().corpus_level_fn
        self.assertAlmostEqual(fns["acc"]([1.0, float("nan"), 0.0, 1.0]), 2 / 3)
        self.assertAlmostEqual(fns["skipped"]([1.0, 0.0, 0.0, 0.0]), 0.25)
        self.assertEqual(set(lm.GSM_NAMES), set(fns))

    def test_20_task_configs(self):
        names = sorted(c.name for c in lm.TASKS_TABLE if c.name.startswith("gsm_infinite"))
        self.assertEqual(len(names), 20)
        base = [c for c in lm.TASKS_TABLE if c.name == "gsm_infinite_hard_65536"][0]
        chat = [c for c in lm.TASKS_TABLE if c.name == "gsm_infinite_hard_65536_chat"][0]
        self.assertTrue(base.hf_repo.endswith("xdg/gsm_infinite_hard_64k"))
        self.assertEqual((base.generation_size, base.stop_sequence), (2048, ["\nProblem:"]))
        self.assertEqual((chat.generation_size, chat.stop_sequence), (8192, []))


# -------------------------------------------------------------------------------------------------------------------- MRCR
class MrcrGrader(unittest.TestCase):
    def test_official_grade_cases(self):
        r = "aYooSG8CQg"
        ans = r + "Poem about tapirs: roses are red"
        self.assertEqual(lm.mrcr_grade(ans, ans, r), 1.0)
        self.assertEqual(lm.mrcr_grade("Poem about tapirs: roses are red", ans, r), 0)             # no prefix
        self.assertEqual(lm.mrcr_grade(" " + ans, ans, r), 0)                                        # the official check does not strip
        self.assertEqual(lm.mrcr_grade("", ans, r), 0)
        half = r + "Poem about tapirs"
        v = lm.mrcr_grade(half, ans, r)
        from difflib import SequenceMatcher
        self.assertAlmostEqual(v, SequenceMatcher(None, "Poem about tapirs", "Poem about tapirs: roses are red").ratio())
        self.assertTrue(0 < v < 1)
        self.assertEqual(lm.mrcr_grade(r, r, r), 1.0)                                                # an empty body against an empty answer

    def test_prefix_is_removed_once_from_both(self):
        r = "ab"
        self.assertEqual(lm.mrcr_grade("abab", "abab", r), 1.0)
        self.assertEqual(lm.mrcr_grade("abXY", "abab", r), SequenceMatcher_ratio("XY", "ab"))


def SequenceMatcher_ratio(a, b):
    from difflib import SequenceMatcher
    return float(SequenceMatcher(None, a, b).ratio())


def mrcr_line(n=2, rs="aYooSG8CQg", answer=None, n_msgs=6):
    msgs = [{"role": "user", "content": "Here are some examples."}, {"role": "user", "content": "Write a poem about tapirs"}, {"role": "assistant", "content": "poem one"},
            {"role": "user", "content": "Write a poem about tapirs"}, {"role": "assistant", "content": "poem two"},
            {"role": "user", "content": f"Prepend {rs} to the 2nd (1 indexed) poem about tapirs. Do not include any other text in your response."}]
    return {"messages": json.dumps(msgs), "answer": answer or rs + "poem two", "random_string_to_prepend": rs, "n_needles": n, "desired_msg_index": 4, "total_messages": 6,
            "n_chars": 500, "bin_upper": 8192}


class Mrcr(unittest.TestCase):
    def test_chat_doc_keeps_the_conversation(self):
        with no_env():
            d = lm.mrcr_doc(mrcr_line(), "t", chat=True)
        self.assertEqual([m["role"] for m in d.specific["chat_messages"]], ["user", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(d.get_golds(), ["aYooSG8CQgpoem two"])
        self.assertEqual((d.specific["n_needles"], d.specific["random_string"]), (2, "aYooSG8CQg"))

    def test_base_rendering_is_a_transcript_ending_in_assistant(self):
        with no_env():
            d = lm.mrcr_doc(mrcr_line(), "t", chat=False)
        self.assertTrue(d.query.startswith("User: Here are some examples.\n\nUser: Write a poem about tapirs\n\nAssistant: poem one\n\n"))
        self.assertTrue(d.query.endswith("in your response.\n\nAssistant:"))
        self.assertNotIn("chat_messages", d.specific)

    def test_skip_when_the_prompt_does_not_fit(self):
        big = mrcr_line()
        msgs = json.loads(big["messages"])
        msgs[2]["content"] = " ".join(["w"] * 3000)
        big["messages"] = json.dumps(msgs)
        with env(2000, 500), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            self.assertTrue(lm.mrcr_doc(big, "t", chat=True).specific["skipped"])
            self.assertTrue(lm.mrcr_doc(big, "t", chat=False).specific["skipped"])
            self.assertFalse(lm.mrcr_doc(mrcr_line(), "t", chat=True).specific["skipped"])

    def test_scores_chat(self):
        with no_env():
            d = lm.mrcr_doc(mrcr_line(n=4), "t", chat=True)
        m = lm.mrcr_metric(base=False)
        s = score(m, d, "aYooSG8CQgpoem two")
        self.assertEqual((s["score"], s["score_4needle"], s["prefix_ok"], s["exact"], s["skipped"]), (1.0, 1.0, 1.0, 1.0, 0.0))
        self.assertTrue(math.isnan(s["score_2needle"]) and math.isnan(s["score_8needle"]))
        w = score(m, d, "aYooSG8CQgpoem one")
        self.assertTrue(0 < w["score"] < 1)
        self.assertEqual((w["prefix_ok"], w["exact"]), (1.0, 0.0))
        z = score(m, d, "poem two")
        self.assertEqual((z["score"], z["prefix_ok"]), (0.0, 0.0))
        self.assertEqual(score(m, d, " aYooSG8CQgpoem two")["score"], 0.0)           # chat: no strip (official)

    def test_base_strips_the_leading_space_of_the_continuation(self):
        with no_env():
            d = lm.mrcr_doc(mrcr_line(), "t", chat=False)
        self.assertEqual(score(lm.mrcr_metric(base=True), d, " aYooSG8CQgpoem two")["score"], 1.0)

    def test_skipped_doc_scores_nan(self):
        d = Doc(task_name="t", query="q", choices=["x"], gold_index=0, specific={"random_string": "r", "n_needles": 2, "skipped": True})
        s = score(lm.mrcr_metric(False), d, "OK")
        self.assertEqual(s["skipped"], 1.0)
        self.assertTrue(all(math.isnan(v) for k, v in s.items() if k != "skipped"))

    def test_task_configs(self):
        names = [c.name for c in lm.TASKS_TABLE if c.name.startswith("mrcr_")]
        self.assertEqual(len(names), 12)
        base = [c for c in lm.TASKS_TABLE if c.name == "mrcr_131072"][0]
        self.assertTrue(base.hf_repo.endswith("xdg/mrcr_131072"))
        self.assertEqual(base.stop_sequence, ["\n\nUser:"])


# ------------------------------------------------------------------------------------------------------------------- FRAMES
class Frames(unittest.TestCase):
    ROW = {"id": 3, "question": "Who?", "answer": "Jane Ballou", "reasoning_types": "Numerical reasoning | Multiple constraints", "n_articles": 2, "article_chars": 10,
           "articles": json.dumps([{"title": "A", "link": "https://en.wikipedia.org/wiki/A", "text": " text a "}, {"title": "B", "link": "https://en.wikipedia.org/wiki/B", "text": "text b"}])}

    def test_prompt_lists_all_articles_then_the_question(self):
        with no_env():
            d = lm.frames_chat_prompt(self.ROW, "t")
        self.assertTrue(d.query.startswith("Answer the question below using the Wikipedia articles provided.\n\n<article 1>\nTitle: A\nURL: https://en.wikipedia.org/wiki/A\n\ntext a\n</article 1>"))
        self.assertIn("<article 2>\nTitle: B", d.query)
        self.assertTrue(d.query.endswith('Question: Who?\n\nGive your final answer on the last line as "Final answer: <answer>".'))
        self.assertEqual((d.specific["gold"], d.specific["question"], d.specific["reasoning_type"], d.specific["truncated"]), ("Jane Ballou", "Who?", "Numerical reasoning", False))

    def test_middle_truncation_keeps_the_question(self):
        row = dict(self.ROW, articles=json.dumps([{"title": "A", "link": "l", "text": " ".join(f"w{i}" for i in range(2000))}]))
        with env(500, 100), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            d = lm.frames_chat_prompt(row, "t")
        self.assertTrue(d.specific["truncated"])
        self.assertTrue(d.query.endswith('"Final answer: <answer>".'))
        self.assertIn("Question: Who?", d.query)

    def test_metric_counts_answered_and_truncated(self):
        d = Doc(task_name="t", query="q", choices=["x"], gold_index=0, specific={"truncated": True})
        s = lm.frames_metric.sample_level_fn.compute(d, ModelResponse(text=["Final answer: x"]))
        self.assertEqual(s, {"answered": 1.0, "truncated": 1.0})
        self.assertEqual(lm.frames_metric.sample_level_fn.compute(d, ModelResponse(text=["  "]))["answered"], 0.0)


# ------------------------------------------------------------------------------------------------------- prompt manager hook
class _TemplateTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + ("<assistant>" if add_generation_prompt else "")


class PromptManagerHook(unittest.TestCase):
    def test_own_conversation_is_rendered_as_given(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}, {"role": "user", "content": "d"}]
        doc = Doc(task_name="t", query="d", choices=["x"], gold_index=0, specific={"chat_messages": msgs})
        out = PromptManager(True, _TemplateTok()).prepare_prompt(doc)
        self.assertEqual(out, "<user>a</user><user>b</user><assistant>c</assistant><user>d</user><assistant>")

    def test_global_system_prompt_only_when_the_conversation_has_none(self):
        doc = Doc(task_name="t", query="d", choices=["x"], gold_index=0, specific={"chat_messages": [{"role": "user", "content": "a"}]})
        self.assertEqual(PromptManager(True, _TemplateTok(), system_prompt="S").prepare_prompt(doc), "<system>S</system><user>a</user><assistant>")
        doc2 = Doc(task_name="t", query="d", choices=["x"], gold_index=0, specific={"chat_messages": [{"role": "system", "content": "own"}, {"role": "user", "content": "a"}]})
        self.assertEqual(PromptManager(True, _TemplateTok(), system_prompt="S").prepare_prompt(doc2), "<system>own</system><user>a</user><assistant>")

    def test_ordinary_docs_are_untouched(self):
        doc = Doc(task_name="t", query="hello", choices=["x"], gold_index=0, specific={"other": 1})
        self.assertEqual(PromptManager(True, _TemplateTok()).prepare_prompt(doc), "<user>hello</user><assistant>")
        doc = Doc(task_name="t", query="hello", choices=["x"], gold_index=0)
        self.assertEqual(PromptManager(True, _TemplateTok()).prepare_prompt(doc), "<user>hello</user><assistant>")
        self.assertEqual(PromptManager(False).prepare_prompt(Doc(task_name="t", query="hello", choices=["x"], gold_index=0, specific={"chat_messages": [{"role": "user", "content": "z"}]})), "hello")


# ------------------------------------------------------------------------------------------------------------- real data (skip)
@unittest.skipUnless((XDG / "gsm_infinite_hard_8k" / "data.parquet").is_file(), "prepared GSM-Infinite data not on disk")
class RealGsmData(unittest.TestCase):
    def test_prepared_rows_are_the_official_sample(self):
        import pandas as pd

        df = pd.read_parquet(XDG / "gsm_infinite_hard_8k" / "data.parquet")
        self.assertEqual(len(df), 19 * 20)
        self.assertEqual(sorted(df.op.unique()), list(range(2, 21)))
        for op, g in df.groupby("op"):
            self.assertEqual(len(g), 20)
            self.assertEqual(g.groupby(["template", "mode"]).size().to_dict(),
                             {("crazy_zootopia", "forwardreverse"): 8, ("crazy_zootopia", "normalforward"): 8, ("movie_festival_awards", "forwardreverse"): 1,
                              ("movie_festival_awards", "normalforward"): 1, ("teachers_in_school", "forwardreverse"): 1, ("teachers_in_school", "normalforward"): 1})
        row = df.iloc[0]
        d = lm.gsm_doc(row.to_dict(), "t", "hard", chat=True)
        self.assertEqual(d.get_golds(), [str(int(row["answer"]))])


@unittest.skipUnless((XDG / "mrcr_8192" / "data.parquet").is_file() and os.path.isdir(QWEN), "prepared MRCR data / tokenizer not on disk")
class RealMrcrData(unittest.TestCase):
    def test_rows_and_grading_of_the_gold(self):
        import pandas as pd

        df = pd.read_parquet(XDG / "mrcr_8192" / "data.parquet")
        self.assertEqual(sorted(df.n_needles.unique()), [2, 4, 8])
        self.assertEqual(df.groupby("n_needles").size().to_dict(), {2: 100, 4: 100, 8: 100})
        for _, r in df.head(30).iterrows():
            self.assertEqual(lm.mrcr_grade(r.answer, r.answer, r.random_string_to_prepend), 1.0)
            msgs = json.loads(r.messages)
            self.assertEqual(msgs[-1]["role"], "user")
            self.assertIn(r.random_string_to_prepend, msgs[-1]["content"])
            self.assertTrue(r.answer.startswith(r.random_string_to_prepend))


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
