"""Tests of the long-context chat task longbench_v2_chat (fork module tasks/tasks/chat_long.py): the official prompt, the middle truncation to the
prompt budget (context - generation - margin, exported by titan_eval as TITAN_EVAL_* variables), the scorer (anchored answer, breakdown buckets) and
the aggregation of the breakdowns (NaN outside a bucket, mean over the members).

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_long.py`. The truncation test reads the Qwen3-1.7B tokenizer when it is on disk
(it skips otherwise); nothing here needs a GPU or the dataset.
"""

import math
import os
import unittest
from unittest import mock

import numpy as np

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_long


TOKENIZER = "/e/project1/trustllm-eu/wang55/torchtitan_assets/Qwen3-1.7B"


def line(context="word " * 50, difficulty="hard", length="long", answer="C", question="What is the answer?"):
    return {
        "_id": "t1", "domain": "Legal", "sub_domain": "Case law", "difficulty": difficulty, "length": length,
        "question": question, "choice_A": "alpha", "choice_B": "beta", "choice_C": "gamma", "choice_D": "delta",
        "answer": answer, "context": context,
    }


class FakeTok:
    """Word-level tokenizer for the truncation rule: one token per whitespace word."""

    def encode(self, text, add_special_tokens=False):
        return text.split(" ")

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def _env(ctx, new):
    return mock.patch.dict(os.environ, {"TITAN_EVAL_MAX_MODEL_LEN": str(ctx), "TITAN_EVAL_MAX_NEW_TOKENS": str(new)}, clear=False)


class PromptTemplate(unittest.TestCase):
    def test_official_template_text(self):
        d = chat_long.longbench_v2_prompt(line(context="  ctx  ", question=" Q? "), "longbench_v2_chat")
        self.assertIn("<text>\nctx\n</text>", d.query)               # context stripped, as pred.py does
        self.assertIn("What is the correct answer to this question: Q?\nChoices:\n(A) alpha\n(B) beta\n(C) gamma\n(D) delta\n", d.query)
        self.assertTrue(d.query.endswith('Format your response as follows: "The correct answer is (insert answer here)".'))
        self.assertEqual(d.choices, ["A", "B", "C", "D"])
        self.assertEqual(d.gold_index, 2)                            # answer C
        self.assertEqual(d.specific["difficulty"], "hard")
        self.assertEqual(d.specific["length"], "long")

    def test_placeholders_replaced_in_official_order(self):
        # pred.py replaces $DOC$ first, then $Q$ and the choices, sequentially: placeholder text inside the context IS replaced later (kept as official)
        d = chat_long.longbench_v2_prompt(line(context="see $Q$ and $C_A$", question="ask"), "t")
        self.assertIn("<text>\nsee ask and alpha\n</text>", d.query)


class Budget(unittest.TestCase):
    def test_budget_is_context_minus_generation_minus_margin(self):
        with _env(40960, 32768):
            self.assertEqual(chat_long.prompt_budget(), 40960 - 32768 - chat_long.CHAT_MARGIN)
            self.assertEqual(chat_long.prompt_budget(), 7936)       # Qwen3-1.7B, thinking budget 32,768

    def test_no_env_means_no_budget(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(chat_long.prompt_budget())

    def test_no_room_is_an_error(self):
        with _env(4096, 4096), mock.patch.object(chat_long, "_tokenizer", return_value=FakeTok()), \
                mock.patch.dict(os.environ, {"TITAN_EVAL_TOKENIZER": "x"}, clear=False):
            with self.assertRaises(ValueError):
                chat_long.longbench_v2_prompt(line(), "t")


class Truncation(unittest.TestCase):
    def test_middle_cut_keeps_head_and_tail_of_budget(self):
        tok = FakeTok()
        text = " ".join(f"w{i}" for i in range(100))
        out = chat_long.truncate_middle(text, 10, tok).split(" ")
        self.assertEqual(out, [f"w{i}" for i in range(5)] + [f"w{i}" for i in range(95, 100)])

    def test_odd_budget_tail_gets_the_extra_token_as_pred_py(self):
        # pred.py: ids[:max//2] + ids[-max//2:] with Python's -max//2 = -ceil(max/2): the tail has ceil(budget/2) tokens
        tok = FakeTok()
        text = " ".join(f"w{i}" for i in range(20))
        out = chat_long.truncate_middle(text, 5, tok).split(" ")
        self.assertEqual(len(out), 5)
        self.assertEqual(out[:2], ["w0", "w1"])
        self.assertEqual(out[2:], ["w17", "w18", "w19"])

    def test_short_text_untouched(self):
        self.assertEqual(chat_long.truncate_middle("a b c", 10, FakeTok()), "a b c")

    def test_prompt_is_cut_to_budget_and_keeps_question(self):
        with _env(4000, 3000), mock.patch.object(chat_long, "_tokenizer", return_value=FakeTok()), \
                mock.patch.dict(os.environ, {"TITAN_EVAL_TOKENIZER": "x"}, clear=False):
            d = chat_long.longbench_v2_prompt(line(context=" ".join(f"w{i}" for i in range(5000)), question="Final question?"), "t")
            budget = 4000 - 3000 - chat_long.CHAT_MARGIN
            self.assertLessEqual(len(d.query.split(" ")), budget)
            self.assertTrue(d.query.endswith("Format your response as follows: \"The correct answer is (insert answer here)\"."))
            self.assertIn("Final question?", d.query)                # the question at the end survives the cut
            self.assertIn("<text>\nw0 w1", d.query)                  # and the head of the text

    @unittest.skipUnless(os.path.isdir(TOKENIZER), "Qwen3-1.7B tokenizer not on disk")
    def test_real_tokenizer_cut_within_budget(self):
        with _env(40960, 32768), mock.patch.dict(os.environ, {"TITAN_EVAL_TOKENIZER": TOKENIZER}, clear=False):
            d = chat_long.longbench_v2_prompt(line(context="The court held that the statute applies. " * 4000), "t")
            tok = chat_long._tokenizer(TOKENIZER)
            n = len(tok.encode(d.query, add_special_tokens=False))
            self.assertLessEqual(n, chat_long.prompt_budget() + 16)   # decode / re-encode may move a few tokens
            self.assertGreater(n, chat_long.prompt_budget() - 64)


def _score(doc, text):
    return chat_long._score(doc, text)


def _doc(difficulty="hard", length="long", answer="C"):
    return chat_long.longbench_v2_prompt(line(difficulty=difficulty, length=length, answer=answer), "t")


class Scorer(unittest.TestCase):
    def test_official_format_correct(self):
        s = _score(_doc(), "The correct answer is (C)")
        self.assertEqual(s["acc"], 1.0)
        self.assertEqual(s["acc_unanchored"], 1.0)

    def test_after_think_block_only(self):
        # the pipeline gives the text after the last </think>; a letter inside the thought must not count (the scorer sees only the text it is given)
        self.assertEqual(_score(_doc(), "")["acc"], 0.0)
        self.assertEqual(_score(_doc(), "Answer: C")["acc"], 1.0)

    def test_wrong_letter(self):
        s = _score(_doc(answer="C"), "The correct answer is (A)")
        self.assertEqual(s["acc"], 0.0)

    def test_hedge_is_no_answer(self):
        s = _score(_doc(answer="C"), "The correct answer is (B) or (C)")
        self.assertEqual(s["acc"], 0.0)

    def test_unanchored_letter_counts_only_as_diagnostic(self):
        s = _score(_doc(answer="C"), "I believe it is C.")
        self.assertEqual(s["acc"], 0.0)
        self.assertEqual(s["acc_unanchored"], 1.0)

    def test_empty_is_zero(self):
        s = _score(_doc(), "   ")
        self.assertEqual(s["acc"], 0.0)
        self.assertEqual(s["acc_unanchored"], 0.0)

    def test_buckets_are_nan_outside_and_value_inside(self):
        s = _score(_doc(difficulty="easy", length="medium", answer="C"), "The correct answer is (C)")
        self.assertEqual(s["acc_easy"], 1.0)
        self.assertTrue(math.isnan(s["acc_hard"]))
        self.assertEqual(s["acc_medium"], 1.0)
        self.assertTrue(math.isnan(s["acc_short"]))
        self.assertTrue(math.isnan(s["acc_long"]))
        w = _score(_doc(difficulty="easy", length="medium", answer="C"), "The correct answer is (A)")
        self.assertEqual(w["acc_easy"], 0.0)
        self.assertEqual(w["acc_medium"], 0.0)


class Aggregation(unittest.TestCase):
    def test_members_average_ignores_non_members(self):
        vals = [1.0, float("nan"), 0.0, float("nan"), 1.0]
        self.assertAlmostEqual(chat_long._members_avg(vals), 2 / 3)
        self.assertTrue(math.isnan(chat_long._members_avg([float("nan")])))

    def test_metric_grouping_names_and_corpus_functions(self):
        m = chat_long.longbench_v2_metric()
        self.assertEqual(list(m.metric_name), list(chat_long.NAMES))
        self.assertIs(m.corpus_level_fn["acc"], np.mean)
        self.assertIs(m.corpus_level_fn["acc_easy"], chat_long._members_avg)
        self.assertNotIn("mean", chat_long._members_avg.__name__)      # lighteval bootstraps a non-"mean" aggregation (NaN-safe)

    def test_sample_level_compute_on_model_response(self):
        m = chat_long.longbench_v2_metric()
        doc = _doc(difficulty="hard", length="long", answer="C")
        out = m.sample_level_fn.compute(doc, ModelResponse(text=["The correct answer is (C)"]))
        self.assertEqual(out["acc"], 1.0)
        self.assertEqual(out["acc_hard"], 1.0)
        self.assertTrue(math.isnan(out["acc_easy"]))
        self.assertFalse(any(k.startswith("pass@") for k in out))       # one sample: no pass@k key

    def test_task_table(self):
        task = chat_long.TASKS_TABLE[0]
        self.assertEqual(task.name, "longbench_v2_chat")
        self.assertEqual(task.hf_repo, "zai-org/LongBench-v2")
        self.assertEqual(task.hf_revision, "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9")
        self.assertEqual(task.generation_size, 32768)
        self.assertEqual(task.stop_sequence, [])


def lbp_line(secondary="T3.2 Single-Hop Fact QA", answer=("Paris",), language="English", difficulty="Hard", token_length="32k"):
    return {
        "id": "p1", "context": "ctx " * 20, "language": language, "token_length": token_length, "primary_task": "T3. Evidence-Grounded QA",
        "secondary_task": secondary, "contextual_requirement": "Full", "question_nonthinking": "NONTHINK Q", "question_thinking": "THINK Q",
        "answer": list(answer), "difficulty": difficulty,
    }


class LongBenchProMetrics(unittest.TestCase):
    def test_accuracy_uses_first_answer_line_after_answer_tag(self):
        self.assertEqual(chat_long.lbp_accuracy(["Paris"], "[Answer] Paris"), 1.0)
        self.assertEqual(chat_long.lbp_accuracy(["Paris"], "reasoning\n[Answer]\nParis"), 1.0)
        self.assertEqual(chat_long.lbp_accuracy(["Paris"], "[answer] Paris"), 0.0)   # the tag is case-sensitive in the official code
        self.assertEqual(chat_long.lbp_accuracy(["Paris"], "Lyon"), 0.0)

    def test_f1_and_subem(self):
        self.assertAlmostEqual(chat_long.lbp_f1(["a", "b"], "a\nz"), 0.5)
        self.assertEqual(chat_long.lbp_subem(["a", "b"], "a\nb"), 1.0)
        self.assertEqual(chat_long.lbp_subem(["a", "b"], "a\nz"), 0.5)

    def test_pairwise_order(self):
        self.assertEqual(chat_long.lbp_pairwise(["a", "b", "c"], "a\nb\nc"), 1.0)
        self.assertEqual(chat_long.lbp_pairwise(["a", "b", "c"], "c\nb\na"), 0.0)

    def test_ndcg_perfect_and_partial(self):
        self.assertAlmostEqual(chat_long.lbp_ndcg(["a", "b", "c"], "a\nb\nc"), 1.0)
        self.assertLess(chat_long.lbp_ndcg(["a", "b", "c"], "c\nx\nb"), 1.0)
        self.assertEqual(chat_long.lbp_ndcg(["a", "b"], "x\ny"), 0.0)

    def test_summary_needs_the_embedding_model(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(chat_long, "_EMBEDDER", None):
            with self.assertRaises(RuntimeError):
                chat_long.lbp_summary(["the answer"], "an answer", False)

    def test_metric_value_empty_is_zero(self):
        self.assertEqual(chat_long.lbp_metric_value("T3.2 Single-Hop Fact QA", ["x"], "", False), 0.0)

    def test_thinking_switch_selects_question(self):
        with mock.patch.dict(os.environ, {chat_long.ENV_THINKING: "0"}):
            self.assertEqual(chat_long.lbp_prompt(lbp_line(), "t").query, "ctx " * 20 + "\n\n\n\nNONTHINK Q")
        with mock.patch.dict(os.environ, {chat_long.ENV_THINKING: "1"}):
            self.assertTrue(chat_long.lbp_prompt(lbp_line(), "t").query.endswith("THINK Q"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(chat_long.lbp_prompt(lbp_line(), "t").query.endswith("THINK Q"))   # unset = thinking (Qwen3 default)

    def test_breakdowns_and_names(self):
        self.assertEqual(len(chat_long.LBP_NAMES), 1 + 4 + 6 + 2 + 2 + 11)
        doc = chat_long.lbp_prompt(lbp_line(answer=("Paris",), language="Chinese", token_length="8k"), "t")
        doc.specific["answer"] = ["Paris"]
        s = chat_long._lbp_score(doc, "Paris")
        self.assertEqual(s["acc"], 1.0)
        self.assertEqual(s["acc_chinese"], 1.0)
        self.assertTrue(math.isnan(s["acc_english"]))
        self.assertEqual(s["acc_8k"], 1.0)
        self.assertTrue(math.isnan(s["acc_256k"]))
        self.assertEqual(s["acc_t3"], 1.0)
        self.assertEqual(s["acc_hard"], 1.0)
        self.assertTrue(math.isnan(s["acc_easy"]))

    def test_task_table_has_both(self):
        names = [t.name for t in chat_long.TASKS_TABLE]
        self.assertEqual(names, ["longbench_v2_chat", "longbench_pro_chat"])


if __name__ == "__main__":
    unittest.main()
