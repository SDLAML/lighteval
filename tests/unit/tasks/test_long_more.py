"""Tests of the long-context tasks of the fork module tasks/tasks/long_more.py: LongBench v2 base, GSM-Infinite (base + chat + zero-noise + symbolic), MRCR (base + chat), FRAMES (chat), and the
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

    def test_base_item_that_does_not_fit_is_skipped_with_a_tiny_prompt(self):
        """BASE tasks (unchanged): skipped = NaN, counted in `skipped`."""
        long_line = gsm_line(problem=" ".join(["w"] * 5000))
        with env(4000, 1000), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            d = lm.gsm_doc(long_line, "t", "hard", chat=False)
            d2 = lm.gsm_doc(gsm_line(), "t", "hard", chat=False)
        self.assertTrue(d.specific["skipped"])
        self.assertEqual(d.query, lm.SKIP_USER)
        self.assertFalse(d2.specific["skipped"])
        self.assertNotIn("over_context", d.specific)

    def test_chat_item_that_does_not_fit_is_not_generated_and_scores_zero(self):
        """CHAT tasks (owner 11-16, `over_context: zero`, the OLMES rule): budget = min(cap, context - prompt - 64); below the floor 1,024 no generation, else generated with the smaller budget."""
        fake = {"ctx": 20000, "cap": 8192}
        small, mid, long = gsm_line(problem="w " * 100), gsm_line(problem=" ".join(["w"] * 12000)), gsm_line(problem=" ".join(["w"] * 19500))
        with env(fake["ctx"], fake["cap"]), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            d_small, d_mid, d_long = (lm.gsm_doc(x, "t", "hard", chat=True) for x in (small, mid, long))
        # the prompt counts words + 6 per message (3 messages): small ~ 120, mid ~ 12,040, long ~ 19,540
        self.assertEqual((d_small.specific["over_context"], d_small.specific["reduced_budget"], d_small.specific["answer_budget"]), (False, False, 8192))
        self.assertNotIn("no_generation", d_small.specific)
        self.assertEqual(d_small.specific["chat_messages"][0]["role"], "system")
        self.assertEqual((d_mid.specific["over_context"], d_mid.specific["reduced_budget"]), (False, True))
        self.assertEqual(d_mid.specific["max_new_tokens"], d_mid.specific["answer_budget"])
        self.assertEqual(d_mid.specific["answer_budget"], 20000 - d_mid.specific["n_tokens"] - lm.GSM_MARGIN)
        self.assertGreaterEqual(d_mid.specific["answer_budget"], lm.GSM_MIN_ANSWER)
        self.assertTrue(d_long.specific["over_context"] and d_long.specific["no_generation"] and d_long.specific["answer_budget"] == 0)
        self.assertEqual(d_long.specific["chat_messages"], [{"role": "user", "content": lm.SKIP_USER}])
        self.assertNotIn("skipped", d_long.specific)

    def test_no_env_never_skips_or_marks(self):
        with no_env():
            d = lm.gsm_doc(gsm_line(problem=" ".join(["w"] * 50000)), "t", "hard", chat=False)
            c = lm.gsm_doc(gsm_line(problem=" ".join(["w"] * 50000)), "t", "hard", chat=True)
        self.assertFalse(d.specific["skipped"])
        self.assertFalse(c.specific["over_context"] or c.specific["reduced_budget"])
        self.assertNotIn("no_generation", c.specific)

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

    def test_26_task_configs(self):
        """20 as built (hard / medium x 8k .. 128k, base + chat) + the 2 zero-noise chat tasks + the 4 symbolic chat tasks (owner 11-16)."""
        names = sorted(c.name for c in lm.TASKS_TABLE if c.name.startswith("gsm_infinite"))
        self.assertEqual(len(names), 26)
        self.assertEqual(sorted(n for n in names if "symbolic" in n), sorted(f"gsm_infinite_symbolic_{n}_chat" for n in lm.GSM_SYMBOLIC_OPS))
        self.assertEqual(sorted(n for n in names if n.endswith("_0_chat") and "symbolic" not in n), ["gsm_infinite_hard_0_chat", "gsm_infinite_medium_0_chat"])
        by = {c.name: c for c in lm.TASKS_TABLE}
        base, chat = by["gsm_infinite_hard_65536"], by["gsm_infinite_hard_65536_chat"]
        self.assertTrue(base.hf_repo.endswith("xdg/gsm_infinite_hard_64k"))
        self.assertEqual((base.generation_size, base.stop_sequence), (4096, ["\nProblem:"]))
        self.assertEqual((chat.generation_size, chat.stop_sequence), (8192, []))
        self.assertEqual((base.version, chat.version), (0, 1), "the base tasks are unchanged; the chat tasks are version 1 (official-setup alignment, 11-16)")
        self.assertTrue(all(by[n].version == 1 for n in names if n.endswith("_chat")) and all(by[n].version == 0 for n in names if not n.endswith("_chat")))
        self.assertTrue(by["gsm_infinite_medium_0_chat"].hf_repo.endswith("xdg/gsm_infinite_medium_0") and by["gsm_infinite_symbolic_8192_chat"].hf_repo.endswith("xdg/gsm_infinite_symbolic_8k"))
        self.assertEqual(by["gsm_infinite_hard_8192_chat"].metrics[0].metric_name, list(lm.gsm_chat_names(range(2, 31))))
        self.assertEqual(by["gsm_infinite_hard_131072_chat"].metrics[0].metric_name, list(lm.gsm_chat_names(range(2, 21))))
        self.assertEqual(by["gsm_infinite_hard_0_chat"].metrics[0].metric_name, list(lm.gsm_chat_names(range(2, 31))))
        self.assertEqual(by["gsm_infinite_hard_8192"].metrics[0].metric_name, list(lm.GSM_NAMES), "the base metric is unchanged")
        self.assertEqual(by["gsm_infinite_symbolic_0_chat"].metrics[0].metric_name, ["answered", "over_context", "reduced_budget"])
        self.assertEqual((by["gsm_infinite_symbolic_0_chat"].generation_size, by["gsm_infinite_symbolic_0_chat"].stop_sequence), (8192, []))
        self.assertIsNone(by["gsm_infinite_hard_8192_chat"].hf_filter, "the chat tasks read the whole prepared file (no op filter)")
        self.assertIsNotNone(by["gsm_infinite_hard_8192"].hf_filter, "the base tasks drop the rows above op 20 of the shared file")
        self.assertEqual(by["gsm_infinite_hard_8192"].hf_filter.max_op, 20)
        self.assertEqual({n: tuple(v) for n, v in lm.GSM_SYMBOLIC_OPS.items()}, {0: (1, 50), 8192: (1, 30), 16384: (1, 20), 32768: (1, 10)})


def chat_gsm_doc(op, over=False, reduced=False, mode="normalforward", answer=6):
    spec = {"op": op, "mode": mode, "over_context": over, "reduced_budget": reduced}
    return Doc(task_name="t", query="q", choices=[str(answer)], gold_index=0, specific=spec)


class GsmChatScoring(unittest.TestCase):
    """The chat metric (owner 11-16): over-context items score 0 and stay in the denominator (official: a failed request becomes an "Error: ..." reply, parsed as wrong), `no_answer` counts generated
    replies only, the official AUC next to `acc`."""

    OPS = tuple(range(2, 31))

    def test_names_and_aggregations(self):
        m = lm.gsm_chat_metric(self.OPS)
        self.assertEqual(m.metric_name[:7], ["acc", "auc", "acc_forward", "acc_reverse", "no_answer", "over_context", "reduced_budget"])
        self.assertEqual(len(m.metric_name), 7 + 29)
        self.assertEqual(set(m.corpus_level_fn), set(m.metric_name))
        self.assertTrue(m.higher_is_better["acc"] and m.higher_is_better["auc"] and not m.higher_is_better["no_answer"] and not m.higher_is_better["over_context"])

    def test_correct_wrong_missing_and_zero(self):
        m = lm.gsm_chat_metric(self.OPS)
        s = score(m, chat_gsm_doc(5), "Define A as 3. Answer: 6.")
        self.assertEqual((s["acc"], s["acc_op5"], s["acc_forward"], s["no_answer"], s["over_context"], s["reduced_budget"]), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
        self.assertEqual(s["auc"], 5.5)
        self.assertTrue(math.isnan(s["acc_op4"]) and math.isnan(s["acc_reverse"]))
        w = score(m, chat_gsm_doc(25, mode="forwardreverse"), "Answer: 7.")
        self.assertEqual((w["acc"], w["acc_op25"], w["acc_reverse"], w["no_answer"], w["auc"]), (0.0, 0.0, 0.0, 0.0, 25.0))
        self.assertEqual(score(m, chat_gsm_doc(5), "I cannot say")["no_answer"], 1.0)
        z = score(m, chat_gsm_doc(30, over=True), "")
        self.assertEqual((z["acc"], z["acc_op30"], z["acc_forward"], z["over_context"], z["auc"]), (0.0, 0.0, 0.0, 1.0, 30.0), "wrong, a member of its op and of the denominator")
        self.assertTrue(math.isnan(z["no_answer"]), "no reply: `no_answer` is the share of GENERATED replies without an integer")
        r = score(m, chat_gsm_doc(6, reduced=True), "Answer: 6.")
        self.assertEqual((r["reduced_budget"], r["acc"]), (1.0, 1.0))
        o = score(m, chat_gsm_doc(6, over=True, reduced=True), "")
        self.assertEqual(o["reduced_budget"], 0.0, "an over-context item is never counted as generated with a reduced budget")

    def test_over_context_items_stay_in_the_mean_and_the_per_op_accuracy(self):
        fns = lm.gsm_chat_metric(self.OPS).corpus_level_fn
        self.assertAlmostEqual(fns["acc"]([1.0, 0.0, 0.0, 1.0]), 0.5)                      # a zeroed item is a 0.0 in the list, not a NaN
        self.assertAlmostEqual(fns["acc_op7"]([float("nan"), 1.0, 0.0, 0.0]), 1 / 3)
        self.assertAlmostEqual(fns["over_context"]([1.0, 0.0, 0.0, 0.0]), 0.25)
        self.assertAlmostEqual(fns["no_answer"]([float("nan"), 1.0, 0.0, 0.0]), 1 / 3)                 # a zeroed item has no reply: not a member of `no_answer`

    def test_auc_is_the_official_trapezoid_of_the_per_op_accuracy(self):
        """gsm-infinite/app.py:65-73 `calculate_auc`: sorted x, np.trapezoid(y, x) * 100; x = the op counts present, y = the accuracy per op (weighted mean over the op's questions)."""
        rng = np.random.default_rng(0)
        for ops in (range(2, 31), range(2, 21), [2, 3, 5, 9], [4]):
            vals, per_op = [], {}
            for op in ops:
                n = int(rng.integers(1, 40))
                ok = (rng.random(n) < (1.0 - op / 40)).astype(float)
                per_op[op] = ok.mean()
                vals += [lm.gsm_auc_code(op, k) for k in ok]
            rng.shuffle(vals)                                                               # the order of the items does not matter
            xs = sorted(per_op)
            want = 0.0 if len(xs) < 2 else float(np.trapezoid([per_op[x] for x in xs], xs) * 100)
            self.assertAlmostEqual(lm.gsm_auc(vals), want, places=9, msg=str(list(ops)))
        # exact values: all correct 2..30 = 2800 (the maximum), all wrong = 0, a straight line from 1 to 0 = 1400
        self.assertAlmostEqual(lm.gsm_auc([lm.gsm_auc_code(o, 1.0) for o in range(2, 31) for _ in range(3)]), 2800.0)
        self.assertEqual(lm.gsm_auc([lm.gsm_auc_code(o, 0.0) for o in range(2, 31)]), 0.0)
        self.assertAlmostEqual(lm.gsm_auc([lm.gsm_auc_code(o, 1.0 if o <= 16 else 0.0) for o in range(2, 31) for _ in range(1)]), (14 + 0.5) * 100, places=6)
        self.assertEqual(lm.gsm_auc([lm.gsm_auc_code(7, 1.0)] * 5), 0.0, "one op count: no area (the official returns 0 for fewer than two points)")
        self.assertTrue(math.isnan(lm.gsm_auc([float("nan")])))
        self.assertAlmostEqual(lm.gsm_auc([2.5, 2.0, 3.5, 3.5, 4.0, 4.0]), 125.0)

    def test_auc_with_several_samples_per_item(self):
        """`ChatSamples` averages the per-sample values of an item: op + 0.5 x (share correct); the decoded per-op accuracy is the mean share (the official `correct_num / len(replies)`)."""
        m = lm.gsm_chat_metric(range(2, 4))
        with no_env():
            d = chat_gsm_doc(2)
            out = m.sample_level_fn.compute(d, ModelResponse(text=["Answer: 6.", "Answer: 1.", "Answer: 6.", "Answer: 6."]))
        self.assertEqual(out["acc"], 0.75)
        self.assertEqual(out["auc"], 2.375)
        self.assertAlmostEqual(lm.gsm_auc([out["auc"], lm.gsm_auc_code(3, 1.0)]), ((0.75 + 1.0) / 2) * 100)

    def test_the_sample_level_function_returns_exactly_the_declared_names(self):
        m = lm.gsm_chat_metric(range(2, 21))
        out = m.sample_level_fn.compute(chat_gsm_doc(9), ModelResponse(text=["Answer: 6."]))
        self.assertEqual(set(out), set(m.metric_name))


class GsmOpFilter(unittest.TestCase):
    """The base tasks read the chat tasks' (larger) files: `GsmLimit(max_op=20)` keeps them at ops 2 .. 20 exactly as built."""

    def rows(self):
        out = []
        for op in range(2, 31):
            for template, mode, n in (("crazy_zootopia", "normalforward", 80), ("movie_festival_awards", "normalforward", 10), ("teachers_in_school", "normalforward", 10),
                                      ("crazy_zootopia", "forwardreverse", 80), ("movie_festival_awards", "forwardreverse", 10), ("teachers_in_school", "forwardreverse", 10)):
                out += [{"op": op, "template": template, "mode": mode, "k": i} for i in range(n)]
        return out

    def test_max_op_only(self):
        rows = self.rows()
        keep = lm.GsmLimit(None, 20).pick(rows)
        self.assertEqual(sorted({rows[i]["op"] for i in keep}), list(range(2, 21)))
        self.assertEqual(len(keep), 19 * 200)
        self.assertEqual(keep, [i for i, r in enumerate(rows) if r["op"] <= 20], "no reordering, no other row dropped")

    def test_max_op_with_the_limit_flag(self):
        rows = self.rows()
        keep = lm.GsmLimit(20, 20).pick(rows)
        self.assertEqual(len(keep), 19 * 20)
        self.assertEqual(keep, [i for i in lm.GsmLimit(20).pick(rows) if rows[i]["op"] <= 20])

    def test_default_filter_function(self):
        with mock.patch.dict(os.environ, {"TITAN_EVAL_SAMPLE_PER_GROUP": ""}):
            self.assertIsNone(lm.gsm_limit_filter())
            self.assertEqual((lm.gsm_limit_filter(20).limit, lm.gsm_limit_filter(20).max_op), (None, 20))
        with mock.patch.dict(os.environ, {"TITAN_EVAL_SAMPLE_PER_GROUP": "40"}):
            f = lm.gsm_limit_filter()
            self.assertEqual((f.limit, f.max_op), (40, None))
            self.assertEqual((lm.gsm_limit_filter(20).limit, lm.gsm_limit_filter(20).max_op), (40, 20))


def symbolic_line(op=3, answer_list=("V4", "V9"), answer_q=5, problem="<context>\nassign V4 = 5.\n</context>\n\nWhich variables equal 5?"):
    return {"id": f"symbolic_8k_op{op}_000", "src_id": 1, "op": op, "n": 7, "length": 8000, "d": 2, "answer_q": answer_q, "answer_list": json.dumps(list(answer_list)), "question": "Which variables equal 5?",
            "solution": "x", "messages": json.dumps([{"role": "user", "content": problem}]), "n_chars": len(problem)}


class GsmSymbolic(unittest.TestCase):
    def test_doc_carries_the_messages_the_gold_list_and_the_value(self):
        with no_env():
            d = lm.gsm_symbolic_doc(symbolic_line(), "t")
        self.assertEqual(d.specific["chat_messages"], [{"role": "user", "content": symbolic_line()["messages"] and json.loads(symbolic_line()["messages"])[0]["content"]}])
        self.assertEqual(d.specific["gold"], ["V4", "V9"])
        self.assertEqual((d.specific["op"], d.specific["answer_q"], d.specific["question"]), (3, 5, "Which variables equal 5?"))
        self.assertEqual(d.choices, ["V4, V9"])
        self.assertFalse(d.specific["over_context"])

    def test_over_context_item_is_not_generated(self):
        long = symbolic_line(problem=" ".join(["w"] * 20000))
        with env(8000, 8192), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            d = lm.gsm_symbolic_doc(long, "t")
        self.assertTrue(d.specific["over_context"] and d.specific["no_generation"])
        self.assertEqual(d.specific["chat_messages"], [{"role": "user", "content": lm.SKIP_USER}])
        self.assertEqual(d.specific["gold"], ["V4", "V9"], "the gold stays: the judge stage counts the item wrong without a call")

    def test_lighteval_metric_is_the_generation_stage_only(self):
        c = [x for x in lm.TASKS_TABLE if x.name == "gsm_infinite_symbolic_16384_chat"][0]
        names = c.metrics[0].metric_name
        self.assertEqual(names, ["answered", "over_context", "reduced_budget"])
        d = lm.gsm_symbolic_doc(symbolic_line(), "t")
        s = c.metrics[0].sample_level_fn.compute(d, ModelResponse(text=["V4, V9"]))
        self.assertEqual((s["answered"], s["over_context"]), (1.0, 0.0))


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

    def test_base_skips_and_chat_shrinks_the_budget_or_marks_over_context(self):
        """Base: the official skip rule (NaN). Chat (owner 11-16, the OLMES rule): budget = min(task budget, context - prompt - 64); below the floor 1,024 the question is not generated."""
        msgs = json.loads(mrcr_line()["messages"])
        msgs[2]["content"] = " ".join(["w"] * 3000)
        big = dict(mrcr_line(), messages=json.dumps(msgs))
        with env(2000, 500), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            b = lm.mrcr_doc(big, "t", chat=False)
            self.assertTrue(b.specific["skipped"] and "over_context" not in b.specific and "no_generation" not in b.specific, "base: the official skip rule, as before")
        n = None
        with env(100000, 5000), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            n = lm.mrcr_doc(big, "t", chat=True).specific["n_tokens"]
        self.assertTrue(n > 3000)
        for ctx, new, want in ((n + 5000 + 64, 5000, ("full", 5000)),                    # exactly fits
                               (n + 5000 + 64 - 1, 5000, ("reduced", 4999)),             # one token short: a reduced budget (still >= 1,024)
                               (n + 1024 + 64, 5000, ("reduced", 1024)),                 # the floor
                               (n + 1024 + 64 - 1, 5000, ("over", 0))):                  # below the floor: not generated
            with env(ctx, new), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
                c = lm.mrcr_doc(big, "t", chat=True)
            sp = c.specific
            kind = "over" if sp["over_context"] else ("reduced" if sp["reduced_budget"] else "full")
            self.assertEqual((kind, sp["answer_budget"]), want, (ctx, new))
            self.assertEqual(sp.get("max_new_tokens"), want[1] if kind == "reduced" else None, "the request's own cap only for a reduced item")
            self.assertEqual(sp["task_budget"], new)
            self.assertEqual(bool(sp.get("no_generation")), kind == "over")
            self.assertNotIn("skipped", sp)
            if kind == "over":
                self.assertEqual((c.query, c.specific["chat_messages"]), (lm.SKIP_USER, [{"role": "user", "content": lm.SKIP_USER}]))
            else:
                self.assertEqual(c.specific["chat_messages"], msgs)

    def test_scores_chat(self):
        with no_env():
            d = lm.mrcr_doc(mrcr_line(n=4), "t", chat=True)
        m = lm.mrcr_metric(base=False)
        s = score(m, d, "aYooSG8CQgpoem two")
        self.assertEqual((s["score"], s["score_4needle"], s["prefix_ok"], s["exact"], s["over_context"], s["reduced_budget"]), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
        self.assertNotIn("skipped", s)
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

    def test_skipped_doc_scores_nan_in_the_base_task(self):
        d = Doc(task_name="t", query="q", choices=["x"], gold_index=0, specific={"random_string": "r", "n_needles": 2, "skipped": True})
        s = score(lm.mrcr_metric(True), d, "OK")
        self.assertEqual(s["skipped"], 1.0)
        self.assertTrue(all(math.isnan(v) for k, v in s.items() if k != "skipped"))

    def test_over_context_doc_scores_zero_and_stays_in_the_mean_of_the_chat_task(self):
        """Owner 11-16: the chat task counts a question that was not generated as WRONG (0, not NaN): the score is over all 300 questions, also per needle count."""
        over = Doc(task_name="t", query="q", choices=["x"], gold_index=0, specific={"random_string": "r", "n_needles": 4, "over_context": True, "no_generation": True})
        m = lm.mrcr_metric(False)
        s = score(m, over, "r the model would have copied it")                       # whatever text came back is not read
        self.assertEqual((s["score"], s["prefix_ok"], s["exact"], s["over_context"], s["reduced_budget"], s["score_4needle"]), (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        self.assertTrue(math.isnan(s["score_2needle"]) and math.isnan(s["score_8needle"]))
        ok = Doc(task_name="t", query="q", choices=["rpoem"], gold_index=0, specific={"random_string": "r", "n_needles": 4, "over_context": False, "reduced_budget": True})
        fine = score(m, ok, "rpoem")
        self.assertEqual((fine["score"], fine["over_context"], fine["reduced_budget"]), (1.0, 0.0, 1.0), "a reduced-budget item is scored normally and counted as reduced")
        agg = m.corpus_level_fn
        self.assertEqual(agg["score"]([fine["score"], s["score"]]), 0.5, "one of two questions not generated: the mean has them both")
        self.assertEqual(agg["score_4needle"]([fine["score_4needle"], s["score_4needle"], float("nan")]), 0.5)
        self.assertEqual(agg["over_context"]([0.0, 1.0, 0.0, 0.0]), 0.25)
        self.assertEqual(agg["reduced_budget"]([1.0, 0.0]), 0.5)
        self.assertEqual(lm._mrcr_names(chat=True)[-2:], ("over_context", "reduced_budget"))
        self.assertEqual(lm._mrcr_names()[-1], "skipped")

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
        self.assertEqual((d.specific["gold"], d.specific["question"], d.specific["reasoning_type"], d.specific["over_context"]), ("Jane Ballou", "Who?", "Numerical reasoning", False))
        self.assertNotIn("no_generation", d.specific)
        self.assertEqual(d.specific["n_tokens"], -1, "no run variables: nothing counted, nothing marked")

    def _frames_row(self, words):
        return dict(self.ROW, articles=json.dumps([{"title": "A", "link": "l", "text": " ".join(f"w{i}" for i in range(words))}]))

    def test_a_prompt_that_fits_is_kept_whole(self):
        with env(10000, 2000), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            d = lm.frames_chat_prompt(self._frames_row(300), "t")
        sp = d.specific
        self.assertEqual((sp["over_context"], sp["reduced_budget"], sp["answer_budget"], sp["task_budget"]), (False, False, 2000, 2000))
        self.assertTrue(sp["n_tokens"] > 300 and sp["n_tokens"] + 2000 + lm.FRAMES_MARGIN <= 10000)
        self.assertIn("w299", d.query)
        self.assertTrue(d.query.endswith('"Final answer: <answer>".') and "no_generation" not in sp and "max_new_tokens" not in sp)

    def test_the_answer_budget_shrinks_to_what_the_context_leaves_and_zero_below_the_floor(self):
        """Owner 11-16 (the OLMES rule, openai_mrcr.py:41-46): budget = min(task budget, context - prompt - 64); the question is not generated only when that is below 1,024."""
        row = self._frames_row(300)
        with env(100000, 5000), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
            n = lm.frames_chat_prompt(row, "t").specific["n_tokens"]
        self.assertEqual(lm.FRAMES_MARGIN, 64)
        for ctx, want in ((n + 5000 + 64, ("full", 5000)), (n + 5000 + 64 - 1, ("reduced", 4999)), (n + 1024 + 64, ("reduced", 1024)), (n + 1024 + 64 - 1, ("over", 0))):
            with env(ctx, 5000), mock.patch.object(lm, "_tokenizer", return_value=FakeTok()):
                d = lm.frames_chat_prompt(row, "t")
            sp = d.specific
            kind = "over" if sp["over_context"] else ("reduced" if sp["reduced_budget"] else "full")
            self.assertEqual((kind, sp["answer_budget"]), want, ctx)
            self.assertEqual(sp.get("max_new_tokens"), want[1] if kind == "reduced" else None)
            self.assertEqual(bool(sp.get("no_generation")), kind == "over")
            self.assertEqual(d.query == lm.SKIP_USER, kind == "over", "an item that is not generated carries the tiny query, every other item its whole, never cut prompt")
            if kind != "over":
                self.assertIn("w299", d.query)
                self.assertEqual(d.specific["n_tokens"], n)
            self.assertEqual((sp["gold"], sp["question"]), ("Jane Ballou", "Who?"), "the judge stage always gets the question and the gold")

    def test_metric_counts_answered_over_context_and_reduced_budget(self):
        d = Doc(task_name="t", query="q", choices=["x"], gold_index=0, specific={"over_context": False, "reduced_budget": True})
        s = lm.frames_metric.sample_level_fn.compute(d, ModelResponse(text=["Final answer: x"]))
        self.assertEqual(s, {"answered": 1.0, "over_context": 0.0, "reduced_budget": 1.0})
        self.assertEqual(lm.frames_metric.sample_level_fn.compute(d, ModelResponse(text=["  "]))["answered"], 0.0)
        over = Doc(task_name="t", query="q", choices=["x"], gold_index=0, specific={"over_context": True, "no_generation": True})
        self.assertEqual(lm.frames_metric.sample_level_fn.compute(over, ModelResponse(text=[""])), {"answered": 0.0, "over_context": 1.0, "reduced_budget": 0.0})
        self.assertEqual(lm.frames_metric.sample_level_fn.compute(over, ModelResponse(text=["OK"]))["answered"], 0.0, "whatever came back, an item not generated is not answered")


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
        """The prepared file is the official `limit` 200 sample (owner rule 11-14), and the opt-in flag at limit 20 gives the first build's 20 per op."""
        import pandas as pd

        df = pd.read_parquet(XDG / "gsm_infinite_hard_8k" / "data.parquet")
        if df.op.max() == 20:
            self.skipTest("the cache still holds the 11-14 build (ops 2..20); run `titan-eval prepare gsm_infinite`")
        self.assertEqual(len(df), 29 * 200)
        self.assertEqual(sorted(df.op.unique()), list(range(2, 31)))                  # 11-16: the official ops_config of config.sh
        for op, g in df.groupby("op"):
            self.assertEqual(len(g), 200)
            self.assertEqual(g.groupby(["template", "mode"]).size().to_dict(),
                             {("crazy_zootopia", "forwardreverse"): 80, ("crazy_zootopia", "normalforward"): 80, ("movie_festival_awards", "forwardreverse"): 10,
                              ("movie_festival_awards", "normalforward"): 10, ("teachers_in_school", "forwardreverse"): 10, ("teachers_in_school", "normalforward"): 10})
        small = df.iloc[lm.GsmLimit(20).pick(df.to_dict("records"))]
        self.assertEqual(len(small), 29 * 20)
        base = df.iloc[lm.GsmLimit(None, 20).pick(df.to_dict("records"))]
        self.assertEqual((len(base), sorted(base.op.unique())), (19 * 200, list(range(2, 21))), "the base task's view of the file = the 11-14 build")
        for op, g in small.groupby("op"):
            self.assertEqual(g.groupby(["template", "mode"]).size().to_dict(),
                             {("crazy_zootopia", "forwardreverse"): 8, ("crazy_zootopia", "normalforward"): 8, ("movie_festival_awards", "forwardreverse"): 1,
                              ("movie_festival_awards", "normalforward"): 1, ("teachers_in_school", "forwardreverse"): 1, ("teachers_in_school", "normalforward"): 1})
        row = df.iloc[0]
        d = lm.gsm_doc(row.to_dict(), "t", "hard", chat=True)
        self.assertEqual(d.get_golds(), [str(int(row["answer"]))])


def _prepared(name):
    import pandas as pd

    f = XDG / name / "data.parquet"
    return pd.read_parquet(f) if f.is_file() else None


@unittest.skipUnless((XDG / "gsm_infinite_symbolic_0" / "data.parquet").is_file() and (XDG / "gsm_infinite_hard_0" / "data.parquet").is_file(), "prepared zero-noise / symbolic data not on disk")
class RealZeroNoiseAndSymbolicData(unittest.TestCase):
    def test_zero_noise_files(self):
        for sub in ("hard", "medium"):
            df = _prepared(f"gsm_infinite_{sub}_0")
            self.assertEqual((len(df), sorted(df.op.unique())), (29 * 200, list(range(2, 31))))
            self.assertEqual(set(df.groupby("op").size()), {200})
            for op, g in df.groupby("op"):
                self.assertEqual(g.groupby(["template", "mode"]).size().to_dict()[("crazy_zootopia", "normalforward")], 80)
            d = lm.gsm_doc(df.iloc[0].to_dict(), "t", sub, chat=True)
            self.assertEqual([m["role"] for m in d.specific["chat_messages"]], ["system", "user", "user"])
            self.assertTrue(d.specific["chat_messages"][2]["content"].endswith("Solution: "))

    def test_symbolic_files(self):
        for length, (lo, hi) in ((0, (1, 50)), (8192, (1, 30)), (16384, (1, 20)), (32768, (1, 10))):
            df = _prepared(f"gsm_infinite_symbolic_{lm.GSM_LABEL[length]}")
            self.assertEqual((len(df), sorted(df.op.unique())), (100 * (hi - lo + 1), list(range(lo, hi + 1))), length)
            self.assertEqual(set(df.groupby("op").size()), {100})
            for _, row in df.head(40).iterrows():
                d = lm.gsm_symbolic_doc(row.to_dict(), "t")
                msgs = d.specific["chat_messages"]
                self.assertEqual([m["role"] for m in msgs], ["user"])
                self.assertTrue(msgs[0]["content"].startswith("<context>") and "Show your step-by-step reasoning" in msgs[0]["content"])
                self.assertGreaterEqual(len(d.specific["gold"]), 1)
                self.assertTrue(all(v.startswith("V") for v in d.specific["gold"]))


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
