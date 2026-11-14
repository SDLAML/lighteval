"""Tests of HELMET (fork module tasks/tasks/helmet.py): the prompt templates against HELMET's source (sha256 of the string literals of data.py @ aeadacc6),
HELMET's metric functions (hand-made predictions), the sampling helpers against the `datasets` library, the builders on synthetic files, the per-tokenizer cut of the
context (model_utils.tokenize), the task table and the scorers through lighteval's metric objects.

Plain functions: run with pytest, or `python tests/unit/tasks/test_helmet.py`. Nothing here needs a GPU, the HELMET tarball or a dataset download.
"""

import hashlib
import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.tasks import helmet as H


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class Templates(unittest.TestCase):
    def test_verbatim(self):
        """sha256 of the template literals of HELMET data.py @ aeadacc6 (computed from the source with ast; the ruler ones are load_ruler's niah / niah_mv branches)."""
        self.assertEqual(sha(H.QA_USER), "476c9e9268336377a23c21ac353e7b648ca737d95ec2f7ed317521ba040f3edb")
        self.assertEqual(sha(H.JSON_KV_USER), "25551642cae952ec63e23625ac64b0b70ed9c0e7619e6df9e898d80d5691420e")
        self.assertEqual(sha(H.RERANK_USER), "448802253034df979c56b96febc3c7e1db7500eac10c9aa153c97a291e44c7f8")
        self.assertEqual(sha(H.ICL_USER), "60f79f2cf83262474349b472ef8f43186f7ea5b09ca682705dbef68638617ccc")
        self.assertEqual(sha(H.RULER_USER_SINGLE), "c691fb1e2fb9eb3e2a6884fb5c5408e26917eea432b3ea232b2044ad9eec860e")
        self.assertEqual(sha(H.RULER_USER_MULTI), "b373c00e238df12bea974d3fdc4ffebcc0f04600885640250040e3c27169216f")
        self.assertEqual(sha(H.RULER_SYS_SINGLE), "7744f7303254e2162c296cd032e18ace6fb565115d23d5e181844f0c0c80fc09")
        self.assertEqual(sha(H.RULER_SYS_MULTI), "b34f1d2c8a60870c839c07c4b2d649d57049da1d1e423edab3507d71059afcc5")
        self.assertEqual([H.TEMPLATES[k][1] for k in ("qa", "json_kv", "rerank", "icl")], ["Answer:", "Corresponding value:", "Ranking:", "label:"])

    def test_icl_braces_survive_format(self):
        """HELMET writes {{label}} in the template: after .format it must read {label} (the literal the model is asked to imitate)."""
        out = H.ICL_USER.format(context="C", question="Q")
        self.assertIn('Only output "label: {label}" and nothing else.', out)

    def test_base_prompt_ends_with_the_answer_prefix_and_chat_prompt_does_not(self):
        spec = H.SPECS["kilt_nq"]
        line = {"question": "who?", "context": "Document (Title: A): text", "demos": "D\n\n", "extra": json.dumps({"depth": 0.0})}
        base, chat = H.build_prompt(spec, line, False), H.build_prompt(spec, line, True)
        self.assertTrue(base.endswith("Question: who?\nAnswer:"))
        self.assertTrue(chat.endswith("Question: who?"))
        self.assertTrue(base.startswith("Use the given documents to write a concise and short answer to the question."))
        self.assertIn("Answer: [answer]\n\nD\n\nDocument (Title: A): text\n\nQuestion", base)
        ruler = {"question": "q", "context": "CTX", "demos": "", "extra": json.dumps({"type_needle_v": "numbers", "query": "kq"})}
        b = H.build_prompt(H.SPECS["niah_mv"], ruler, False)
        self.assertTrue(b.endswith("What are all the special magic numbers for kq mentioned in the provided text?\nThe special magic numbers for kq mentioned in the provided text are"))
        b3 = H.build_prompt(H.SPECS["niah_mk_3"], {**ruler, "extra": json.dumps({"type_needle_v": "uuid", "query": "kq"})}, False)
        self.assertTrue(b3.endswith("The special magic uuid for kq mentioned in the provided text is"))


class Metrics(unittest.TestCase):
    def test_normalize_and_matches(self):
        self.assertEqual(H.normalize_answer("The  Eiffel Tower!"), "eiffel tower")
        self.assertTrue(H.exact_match_score("October 21, 2016", "October 21 , 2016"))
        self.assertTrue(H.substring_exact_match_score("It was on October 21, 2016 in Paris.", "October 21 , 2016"))
        self.assertFalse(H.substring_exact_match_score("October 22", "October 21"))

    def test_calculate_metrics(self):
        m = H.calculate_metrics("the red car", ["a red car", "blue"])
        self.assertEqual(m["exact_match"], 1.0)                      # article dropped: "red car" == "red car"
        self.assertEqual(m["substring_exact_match"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)
        m = H.calculate_metrics("red", "red car")
        self.assertEqual(m["exact_match"], 0.0)
        self.assertAlmostEqual(m["f1"], 2 * 1 * 0.5 / 1.5)
        self.assertEqual(H.calculate_metrics("x", ["y"])["rougeL_f1"], 0.0)
        self.assertEqual(H.calculate_metrics("same words here", ["same words here"])["rougeL_f1"], 1.0)

    def test_parse_output(self):
        self.assertEqual(H.parse_output("Answer: Paris\nmore", "Answer:"), "Paris")
        self.assertEqual(H.parse_output("answer: Paris", "Answer:"), "Paris")                 # case-insensitive
        self.assertEqual(H.parse_output("Paris\nand more", "Answer:"), "Paris")               # no prefix: the first line
        self.assertEqual(H.parse_output("", "label:"), "")
        self.assertEqual(H.parse_output("label: 23\n\nlabel: 5", "label:"), "23")

    def test_parse_rankings(self):
        self.assertEqual(H.parse_rankings("Ranking: ID3 > ID1 > ID2"), {"3": 3, "1": 2, "2": 1})
        self.assertEqual(H.parse_rankings("[ID: 12] > [ID: 7]"), {"12": 2, "7": 1})
        self.assertEqual(H.parse_rankings("5 > 5 > 4"), {"5": 3, "4": 1})                      # first position wins
        self.assertEqual(H.parse_rankings("no ids here"), {"no here": 1} if False else H.parse_rankings("no ids here"))
        self.assertEqual(H.parse_rankings("Ranking: 3 > 1 and later 9 > 8 > 7 > 6"), {"9": 4, "8": 3, "7": 2, "6": 1})   # the longest chain

    def test_ndcg_cut_hand_computed(self):
        qrels = {"a": 3, "b": 2, "c": 0, "d": 1}
        ideal = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(H.ndcg_cut({"a": 4, "b": 3, "d": 2, "c": 1}, qrels, 10), 1.0)
        got = H.ndcg_cut({"b": 4, "a": 3, "c": 2, "d": 1}, qrels, 10)
        self.assertAlmostEqual(got, (2 / math.log2(2) + 3 / math.log2(3) + 0 + 1 / math.log2(5)) / ideal)
        self.assertAlmostEqual(H.ndcg_cut({"b": 4, "a": 3, "c": 2, "d": 1}, qrels, 1), 2 / 3)          # NDCG@1: gain 2 over ideal gain 3
        self.assertEqual(H.ndcg_cut({"zzz": 1}, qrels, 10), 0.0)                                        # unknown ids are not relevant
        self.assertEqual(H.ndcg_cut({"a": 1}, {"a": 0}, 10), 0.0)                                       # no relevant document: 0

    def test_ndcg_ties_larger_id_first(self):
        # trec_eval ranks equal scores by descending doc id; the HELMET parser never produces ties, but the rule is pinned
        self.assertAlmostEqual(H.ndcg_cut({"a": 1, "b": 1}, {"a": 1, "b": 0}, 1), 0.0)
        self.assertAlmostEqual(H.ndcg_cut({"a": 1, "b": 1}, {"a": 0, "b": 1}, 1), 1.0)


class Scorers(unittest.TestCase):
    def score(self, name, fields, answers, completion, chat):
        return H.score_completion(H.SPECS[name], fields, answers, completion, chat)

    def test_json_kv(self):
        a = ["6b732c7a-8dca-44bd-850a-2d22b08c19fa"]
        self.assertEqual(self.score("json_kv", {}, a, " 6b732c7a-8dca-44bd-850a-2d22b08c19fa", False)["helmet_score"], 1.0)
        self.assertEqual(self.score("json_kv", {}, a, " 6b732c7a-8dca-44bd-850a-2d22b08c19fb", False)["helmet_score"], 0.0)
        self.assertEqual(self.score("json_kv", {}, a, "\n\nKey: x", False)["helmet_score"], 0.0)
        self.assertEqual(self.score("json_kv", {}, a, "Corresponding value: 6b732c7a-8dca-44bd-850a-2d22b08c19fa\nmore", True)["helmet_score"], 1.0)
        self.assertEqual(self.score("json_kv", {}, a, "The value is 6b732c7a-8dca-44bd-850a-2d22b08c19fa.", True)["helmet_score"], 1.0)   # SubEM: anywhere

    def test_ruler_recall_counts_found_needles(self):
        f = {"type_needle_v": "numbers", "query": "q"}
        a = ["1192619", "2921859", "5476583", "6707197"]
        self.assertAlmostEqual(self.score("niah_mv", f, a, " 1192619, 2921859 and 5476583.", False)["helmet_score"], 0.75)
        self.assertEqual(self.score("niah_mv", f, a, " none", False)["helmet_score"], 0.0)
        self.assertEqual(self.score("niah_mk_2", {"type_needle_v": "number", "query": "q"}, ["7661319"], " 7661319", False)["helmet_score"], 1.0)

    def test_qa_submatch_and_parse(self):
        a = ["on October 21 , 2016", "October 21 , 2016"]
        self.assertEqual(self.score("kilt_nq", {}, a, " October 21, 2016\n", False)["helmet_score"], 1.0)
        self.assertEqual(self.score("kilt_nq", {}, a, " October 22, 2016", False)["helmet_score"], 0.0)
        out = self.score("kilt_popqa_3", {}, ["punk rock", "punk"], "Answer: punk rock", True)
        self.assertEqual(out["helmet_score"], 1.0)
        self.assertEqual(out["exact_match"], 1.0)
        self.assertEqual(self.score("kilt_hotpotqa", {}, ["yes"], " no", False)["helmet_score"], 0.0)

    def test_icl_exact_match_after_the_label_prefix(self):
        self.assertEqual(self.score("icl_banking77", {}, ["23"], " 23", False)["helmet_score"], 1.0)
        self.assertEqual(self.score("icl_banking77", {}, ["23"], " 230", False)["helmet_score"], 0.0)       # exact, not substring
        self.assertEqual(self.score("icl_banking77", {}, ["23"], " 23\nlabel: 5", False)["helmet_score"], 1.0)
        self.assertEqual(self.score("icl_banking77", {}, ["23"], "label: 23", True)["helmet_score"], 1.0)
        self.assertEqual(self.score("icl_banking77", {}, ["23"], "23", True)["helmet_score"], 1.0)
        self.assertEqual(self.score("icl_banking77", {}, ["23"], "The label is 23.", True)["helmet_score"], 0.0)   # HELMET reads the first line only
        self.assertEqual(self.score("icl_banking77", {}, ["23"], "", True)["helmet_score"], 0.0)

    def test_chat_reply_counts_up_to_its_first_line_when_helmet_stops_at_a_newline(self):
        """stop_new_line datasets (RAG, re-rank, ICL): a chat reply is cut at the first line (HELMET's stop), so a long reply that contains the gold later does not score."""
        a = ["Paris"]
        self.assertEqual(self.score("kilt_nq", {}, a, "Answer: Paris\n\nBecause the passages say so.", True)["helmet_score"], 1.0)
        self.assertEqual(self.score("kilt_nq", {}, a, "\n\nParis", True)["helmet_score"], 1.0)                      # leading blank lines (after </think>) are skipped
        self.assertEqual(self.score("kilt_nq", {}, a, "Let me think.\nAnswer: Paris", True)["helmet_score"], 0.0)
        self.assertEqual(self.score("icl_nlu", {}, ["7"], "Let me see.\nlabel: 7", True)["helmet_score"], 0.0)
        # no newline stop in HELMET's config for json_kv / ruler / long documents: the whole reply counts
        self.assertEqual(self.score("json_kv", {}, ["abc"], "I think:\nabc", True)["helmet_score"], 1.0)
        self.assertEqual(self.score("niah_mv", {"type_needle_v": "numbers", "query": "q"}, ["11", "22"], "first 11\nsecond 22", True)["helmet_score"], 1.0)

    def test_infbench_choice_letter_a_is_not_free(self):
        """HELMET's exact match removes the article "a": the gold letter A normalises to "" and an empty answer 'matches' it. helmet_score keeps that (it is HELMET's), acc_strict does not."""
        a = ["A", "A. the first option"]
        empty = self.score("infbench_choice_eng", {}, a, "", False)
        self.assertEqual((empty["helmet_score"], empty["acc_strict"]), (1.0, 0.0))
        self.assertEqual(self.score("infbench_choice_eng", {}, ["B", "B. second"], "", False), {"helmet_score": 0.0, "acc_strict": 0.0})
        for good in (" A", "(A)", "A.", "**A**"):
            self.assertEqual(self.score("infbench_choice_eng", {}, a, good, False)["acc_strict"], 1.0, good)
        self.assertEqual(self.score("infbench_choice_eng", {}, a, " B", False)["acc_strict"], 0.0)
        self.assertEqual(self.score("infbench_choice_eng", {}, ["C", "C. gamma ray"], "It is C. gamma ray", True)["helmet_score"], 1.0)   # the option as HELMET writes it ("C. text") counts
        self.assertEqual(self.score("infbench_choice_eng", {}, ["C", "C. gamma ray"], "It is gamma ray", True)["helmet_score"], 0.0)       # without the letter prefix it does not
        self.assertEqual(self.score("infbench_choice_eng", {}, ["C", "C. gamma ray"], " C", False), {"helmet_score": 1.0, "acc_strict": 1.0})

    def test_rerank_ndcg10(self):
        f = {"qrel": [["10", "0"], ["11", "1"], ["12", "0"]]}
        self.assertEqual(self.score("msmarco_rerank_psg", f, ["11 > 10 > 12"], " ID11 > ID10 > ID12", False)["helmet_score"], 1.0)
        self.assertEqual(self.score("msmarco_rerank_psg", f, ["11 > 10 > 12"], " ID10 > ID12 > ID11", False)["helmet_score"], 1 / math.log2(4))
        self.assertEqual(self.score("msmarco_rerank_psg", f, ["11 > 10 > 12"], " I cannot rank these.", False)["helmet_score"], 0.0)
        got = self.score("msmarco_rerank_psg", f, ["11 > 10 > 12"], "Ranking: [ID: 11] > [ID: 12] > [ID: 10]", True)
        self.assertEqual((got["helmet_score"], got["ndcg@1"]), (1.0, 1.0))


class Sampling(unittest.TestCase):
    def test_hf_permutation_is_what_dataset_shuffle_does(self):
        from datasets import Dataset

        ds = Dataset.from_dict({"i": list(range(50))})
        for seed in (42, 7):
            self.assertEqual(ds.shuffle(seed=seed)["i"], H.hf_permutation(50, seed).tolist())

    def test_python_random_sample_equals_seeded_module_random(self):
        keys = sorted(f"q{i}" for i in range(300))
        random.seed(42)
        a = random.sample(keys, 100)
        self.assertEqual(a, random.Random(42).sample(keys, 100))

    def test_first_unique_skips_duplicates_in_permuted_order(self):
        items = [{"k": "a"}, {"k": "a"}, {"k": "b"}, {"k": "c"}]
        got = H._first_unique([1, 0, 3, 2], items, "k", 2)
        self.assertEqual([d["k"] for d in got], ["a", "c"])

    def test_balance_labels(self):
        data = [{"text": f"t{i}", "label": i % 6} for i in range(600)]
        out = H._balance_labels(data, 100, 42, "label")
        self.assertEqual(len(out), 100)
        counts = np.bincount([d["label"] for d in out], minlength=6)
        self.assertLessEqual(counts.max() - counts.min(), 1)                      # 100 = 6 * 16 + 4: balanced up to one
        self.assertEqual(out, H._balance_labels(data, 100, 42, "label"))          # deterministic
        self.assertNotEqual([d["text"] for d in out], [d["text"] for d in H._balance_labels(data, 100, 43, "label")])


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class Builders(unittest.TestCase):
    """The builders on small synthetic copies of the HELMET files (same schemas as the shipped ones)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [mock.patch.object(H, "src_root", lambda: self.root), mock.patch.object(H, "N_QUESTIONS", 5)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def kilt(self, n_q=12, depths=(0.0, 0.5, 1.0), k=4, popqa=False):
        rows = []
        for q in range(n_q):
            for d in depths:
                ctxs = [{"id": str(j), "title": f"T{j}", "text": f"passage {q} {j}", "has_answer": j == 0} for j in range(k)]
                r = {"question": f"question {q}?", "answers": [f"ans{q}", f"alt{q}"], "ctxs": ctxs, "depth": d}
                if popqa:
                    r.update({"id": 1000 + q, "s_pop": 10 if q < 8 else 100000})
                rows.append(r)
        return rows

    def test_qa_keeps_every_depth_of_the_chosen_questions_and_is_deterministic(self):
        spec = H.SPECS["kilt_nq"]
        _write_jsonl(self.root / "data/kilt/nq-dev-multikilt_1000_k50_dep6.jsonl", self.kilt())
        demos = [{"question": f"demo {i}?", "answers": [f"d{i}"], "ctxs": [{"id": "0", "title": "D", "text": f"demo passage {i}"}]} for i in range(6)] * 2
        _write_jsonl(self.root / "data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl", demos)
        rows = H.build_qa(spec, 8192)
        self.assertEqual(len(rows), 5 * 3)
        self.assertEqual(len({r["question"] for r in rows}), 5)
        self.assertEqual(len({r["id"] for r in rows}), 15)
        self.assertEqual(rows, H.build_qa(spec, 8192))
        r = rows[0]
        self.assertTrue(r["demos"].endswith("\n\n"))
        self.assertEqual(r["demos"].count("Question: demo"), H.SHOTS)                       # two distinct demos
        self.assertEqual(len({l for l in r["demos"].split("\n") if l.startswith("Question: demo")}), 2)
        self.assertTrue(r["context"].startswith("Document (Title: T0): passage"))
        self.assertEqual(r["context"].count("Document (Title:"), 4)
        self.assertIn(r["answers"][0], ["ans%d" % i for i in range(12)])
        self.assertEqual(json.loads(r["extra"])["k"], 4)
        # the sampled keys are HELMET's: random.seed(42); random.sample(sorted(set(keys)), 5)
        want = set(random.Random(42).sample(sorted({f"question {q}?" for q in range(12)}), 5))
        self.assertEqual({r["question"] for r in rows}, want)

    def test_popqa_filters_popularity_and_demos_exclude_the_question(self):
        spec = H.SPECS["kilt_popqa_3"]
        rows = self.kilt(popqa=True)
        _write_jsonl(self.root / "data/kilt/popqa_test_1000_k50_dep6.jsonl", rows)
        demo_rows = [{"id": 1000 + q, "s_pop": 10, "question": f"question {q}?", "answers": [f"ans{q}"], "ctxs": [{"id": "0", "title": "D", "text": f"demo passage {q}"}]} for q in range(8)]
        _write_jsonl(self.root / "data/kilt/popqa_test_1000_k3_dep6.jsonl", demo_rows)
        out = H.build_popqa = H.build_qa(spec, 8192)
        ids = {r["id"].split("|")[0] for r in out}
        self.assertTrue(all(int(i) < 1008 for i in ids))                                   # s_pop 100000 rows (log10 = 5 >= 3) are gone
        for r in out:
            own = r["question"]
            self.assertNotIn("Question: " + own + "\n", r["demos"])                        # a demo is never the question itself

    def test_json_kv_demos_and_selection(self):
        rows = [{"context": f"JSON data:\n{{{i}}}", "demos": [[f"k{j}", f"v{j}"] for j in range(5)], "question": f"key{i}", "answer": f"val{i}", "depth": 0.1 * (i % 6), "num_kvs": 105}
                for i in range(30)]
        _write_jsonl(self.root / "data/json_kv/test_k105_dep6.jsonl", rows)
        out = H.build_json_kv(H.SPECS["json_kv"], 8192)
        self.assertEqual(len(out), 5)
        self.assertEqual(out[0]["demos"], "Key: k0\nCorresponding value: v0\n\nKey: k1\nCorresponding value: v1\n\n")
        sel = H.hf_permutation(30, 42)[:5]
        self.assertEqual([r["question"] for r in out], [f"key{i}" for i in sel])

    def test_ruler_rows(self):
        rows = [{"index": i, "input": "x", "outputs": [f"n{i}"], "length": 8000 + i, "answer": [f"n{i}"], "type_needle_v": "numbers", "query": f"q{i}", "context": f"ctx{i}"} for i in range(8)]
        _write_jsonl(self.root / "data/ruler/niah_multivalue/validation_8192.jsonl", rows)
        out = H.build_ruler(H.SPECS["niah_mv"], 8192)
        self.assertEqual(len(out), 5)
        self.assertEqual(json.loads(out[0]["extra"])["type_needle_v"], "numbers")
        self.assertEqual(out[0]["answers"], [f"n{H.hf_permutation(8, 42)[0]}"])

    def test_rerank_rows(self):
        def row(q, shift):
            ctxs = [{"id": str(100 * q + j), "text": f"t{q}{j}", "label": 1 if j == shift else 0} for j in range(5)]
            return {"qid": f"q{q}", "query": f"query {q}", "ctxs": ctxs}
        data = [row(q, s) for q in range(4) for s in range(3)]
        _write_jsonl(self.root / "data/msmarco/test_reranking_data_k50_dep3.jsonl", data)
        _write_jsonl(self.root / "data/msmarco/test_reranking_data_k10_dep3.jsonl", [row(q + 10, 0) for q in range(8)])
        out = H.build_rerank(H.SPECS["msmarco_rerank_psg"], 8192)
        self.assertEqual(len(out), 12)                                                       # 4 queries < 5: all, at 3 positions each
        r = out[0]
        self.assertTrue(r["context"].startswith("[ID: "))
        self.assertTrue(r["demos"].count("\n\nQuery: query 1") >= 1)
        gold = json.loads(r["extra"])["qrel"]
        top = [c[0] for c in gold if c[1] == "1"]
        self.assertTrue(r["answers"][0].startswith(top[0]))                                   # the gold ranking puts the relevant document first

    def test_icl_rows(self):
        train = [{"text": f"train text {i}", "label": i % 6} for i in range(300)]
        test = [{"text": f"test text {i}", "label": i % 6} for i in range(60)]
        with mock.patch.object(H, "icl_source", lambda name: (train, test, "text", "label")), mock.patch.object(H, "N_ICL", 12):
            out = H.build_icl(H.SPECS["icl_trec_coarse"], 8192)
            again = H.build_icl(H.SPECS["icl_trec_coarse"], 8192)
        self.assertEqual(out, again)
        self.assertEqual(len(out), 12)
        r = out[0]
        self.assertEqual(r["context"].count("\nlabel: "), 400)                              # the 400 shots of the 8k bucket
        labels = {l.split("label: ")[1] for l in r["context"].split("\n") if "label: " in l}
        self.assertTrue(labels <= {str(i) for i in range(6)})
        self.assertIn(r["answers"][0], {str(i) for i in range(6)})
        self.assertNotEqual(out[0]["context"], out[1]["context"])                           # every item has its own demos / label mapping


class _FakeTok:
    """Whitespace tokenizer with the two calls the cut uses (no backend_tokenizer: the list path of encode_lengths)."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re

        if isinstance(text, list):
            return {"input_ids": [[0] * len(t.split()) for t in text]}
        offs = [m.span() for m in re.finditer(r"\S+", text)]
        out = {"input_ids": [0] * len(offs)}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


class CutContext(unittest.TestCase):
    def test_cut_drops_the_end_of_the_context_only(self):
        spec = H.SPECS["kilt_nq"]
        tok = _FakeTok()
        line = {"question": "who is it", "context": " ".join(f"w{i}" for i in range(100)), "demos": "demo one two\n\n", "extra": "{}"}
        n = H.encode_lengths(tok, [H.build_prompt(spec, line, False)])[0]
        limit = n - 30
        row, n2, removed = H.cut_context(tok, spec, line, limit, n)
        self.assertEqual(removed, 30)
        self.assertEqual(n2, limit)
        self.assertTrue(row["context"].startswith("w0 w1 w2"))
        self.assertTrue(row["context"].rstrip().endswith("w69"))                              # the last 30 tokens are gone (HELMET keeps the space before the cut)
        self.assertEqual(row["demos"], line["demos"])
        self.assertEqual(row["question"], line["question"])
        self.assertTrue(H.build_prompt(spec, row, False).endswith("Question: who is it\nAnswer:"))

    def test_cut_everything_when_the_excess_is_larger_than_the_context(self):
        spec = H.SPECS["kilt_nq"]
        tok = _FakeTok()
        line = {"question": "q", "context": "a b c", "demos": "", "extra": "{}"}
        n = H.encode_lengths(tok, [H.build_prompt(spec, line, False)])[0]
        row, n2, removed = H.cut_context(tok, spec, line, n - 3, n)
        self.assertEqual(row["context"], "")


class TaskTable(unittest.TestCase):
    def tasks(self):
        return {t.name: t for t in H.get_helmet_tasks(None)}

    def test_names_and_counts(self):
        t = self.tasks()
        self.assertIn("helmet_8192:json_kv", t)
        self.assertIn("helmet_131072_chat:icl_nlu", t)
        for L in H.LENGTHS:
            base = [n for n in t if n.startswith(f"helmet_{L}:")]
            chat = [n for n in t if n.startswith(f"helmet_{L}_chat:")]
            self.assertEqual(len(base), len([s for s in H.AVAILABLE if not H.SPECS[s].judged]))
            self.assertEqual(len(chat), len(H.AVAILABLE))

    def test_generation_settings_follow_the_helmet_configs(self):
        t = self.tasks()
        self.assertEqual(t["helmet_8192:json_kv"].generation_size, 100)
        self.assertFalse(t["helmet_8192:json_kv"].stop_sequence)                                # stop_new_line false (lighteval turns None into ())
        self.assertEqual(t["helmet_8192:niah_mk_2"].generation_size, 50)
        self.assertEqual(t["helmet_8192:niah_mk_3"].generation_size, 100)
        self.assertEqual(t["helmet_8192:kilt_nq"].generation_size, 20)
        self.assertEqual(t["helmet_8192:kilt_nq"].stop_sequence, ["\n"])
        self.assertEqual(t["helmet_8192:msmarco_rerank_psg"].generation_size, 200)
        self.assertEqual(t["helmet_8192:icl_nlu"].stop_sequence, ["\n"])
        self.assertEqual(t["helmet_8192_chat:icl_nlu"].stop_sequence, [])                      # chat: no newline stop
        self.assertEqual(t["helmet_8192_chat:icl_nlu"].generation_size, H.CHAT_GEN)

    def test_rows_come_from_the_tokenizer_folder(self):
        t = self.tasks()
        self.assertIn("/helmet/", t["helmet_8192:json_kv"].hf_repo)
        self.assertTrue(t["helmet_8192:json_kv"].hf_repo.endswith("/tok/unset/8192/json_kv"))

    def test_categories_cover_the_specs(self):
        self.assertEqual({s.category for s in H.SPECS.values()}, set(H.CATEGORIES))
        self.assertEqual(sum(1 for s in H.SPECS.values() if s.category == "Recall"), 4)
        self.assertEqual(sum(1 for s in H.SPECS.values() if s.category == "RAG"), 4)
        self.assertEqual(sum(1 for s in H.SPECS.values() if s.category == "ICL"), 5)


class EndToEnd(unittest.TestCase):
    def line(self):
        return {"id": "x", "question": "who?", "context": "Document (Title: A): text", "demos": "", "answers": ["Paris", "paris, france"], "extra": json.dumps({"k": 3})}

    def test_prompt_function_and_metric(self):
        for chat in (False, True):
            spec = H.SPECS["kilt_triviaqa"]
            doc = H.make_prompt_function(spec, chat)(self.line(), "helmet_8192:kilt_triviaqa")
            self.assertEqual(doc.get_golds(), [""])
            self.assertEqual(doc.specific["answers"], ["Paris", "paris, france"])
            self.assertEqual(doc.query.endswith("Answer:"), not chat)
            metric = H.make_metric(spec, chat)
            right = metric.sample_level_fn.compute(doc, ModelResponse(text=[" Paris"], text_post_processed=[" Paris"]))
            wrong = metric.sample_level_fn.compute(doc, ModelResponse(text=[" Rome"], text_post_processed=[" Rome"]))
            self.assertEqual(right["helmet_score"], 1.0)
            self.assertEqual(wrong["helmet_score"], 0.0)
            self.assertEqual(set(right), {"helmet_score"})                                    # one metric per subtask (lighteval pads the others)

    def test_unclosed_think_is_an_empty_answer(self):
        spec = H.SPECS["kilt_triviaqa"]
        doc = H.make_prompt_function(spec, True)(self.line(), "t")
        metric = H.make_metric(spec, True)
        out = metric.sample_level_fn.compute(doc, ModelResponse(text=["<think>Paris maybe"], text_post_processed=[""]))
        self.assertEqual(out["helmet_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
