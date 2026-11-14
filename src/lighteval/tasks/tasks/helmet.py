# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""HELMET (titan_eval, owner 11-14 "definitely not just RULER"; plan chat_eval 3.7). Module helmet.py of the fork.

HELMET (Yen et al., arXiv 2410.02694, code github.com/princeton-nlp/HELMET @ aeadacc6, MIT; data HF princeton-nlp/HELMET): long-context evaluation in seven
categories at the lengths 8k, 16k, 32k, 64k, 128k. What is built here (everything the rule can score; the model-judged parts are CHAT only):

    Recall    json_kv, niah_mk_2 (RULER multi-key needle), niah_mk_3 (multi-key UUID), niah_mv (multi-value)         SubEM / ruler_recall
    RAG       kilt_nq, kilt_triviaqa, kilt_hotpotqa, kilt_popqa_3 (retrieved passages, gold at 6 / 3 depths)           SubEM
    Re-rank   msmarco_rerank_psg                                                                                      NDCG@10
    ICL       icl_trec_coarse, icl_trec_fine, icl_banking77, icl_clinic150, icl_nlu (many-shot, label ids shuffled)    exact match
    LongQA    infbench_qa_eng (ROUGE-L F1), infbench_choice_eng (accuracy); narrativeqa (GPT-4o judge: chat + local judge)
    Summ      infbench_sum_eng, multi_lexsum (GPT-4o judge on key points: chat + local judge)
    Cite (ALCE: needs google/t5_xxl_true_nli_mixture) is NOT built.

Task names (lighteval): base `helmet_<L>:<name>` and chat `helmet_<L>_chat:<name>`, L in 8192 .. 131072; `helmet_<L>|0` expands to all of its subtasks.
Every subtask reports `helmet_score` = the metric HELMET's collect_results.py takes for that dataset (primary) plus diagnostics. The category score is the mean of
its datasets, HELMET's overall the mean of the categories (registry `subgroups` + `aggregate: group_mean`, titan_eval).

LENGTH. HELMET defines a length in LLAMA-2 TOKENS: its data are cut with the Llama-2 tokenizer (json_kv / RULER / KILT / MS MARCO files ship per length, the
many-shot ICL sets are sized in shots, the long documents are cut by `truncate_llama2`), and the prompt is then cut again at run time, in the MODEL's own tokens, so that
prompt + generation fit the length (model_utils.tokenize: the END of the `context` field is dropped; instruction, demos and question stay). We keep both steps:
the shipped data are the tokenizer-independent "base" data (Llama-2 lengths, as HELMET), and `titan-eval prepare helmet --model X` makes the per-tokenizer copy in which
the context is cut the way HELMET's runner would cut it for that model (RULER-style per-tokenizer preparation; nothing is tokenised at run time). Qwen / o200k vocabularies
need fewer tokens than Llama-2, so fewer rows are cut than in HELMET's own Llama runs; the manifest of the copy says how many rows were cut and by how much.

BASE variant (`helmet_<L>`): HELMET's `prompt_template` = user template + "\n" + the answer prefix ("Answer:", "label:", "Ranking:", ...), plain text, no chat template, greedy,
the generation length of the HELMET config, a newline stop for RAG / re-rank / ICL; the prefix is prepended to the completion before parsing (HELMET eval.py). The
2 in-context demos of HELMET (shots 2) are part of the prepared prompts. CHAT variant (`helmet_<L>_chat`): HELMET's `user_template` (without the answer prefix) as one user
message through the model's chat template; the scored text is what follows the last </think>; no newline stop. Both variants score with the same HELMET functions.

DATA. `titan-eval prepare helmet` (login node) cuts the shipped HELMET files with HELMET's own sampling (random.seed(42) + random.sample of the sorted question keys,
`Dataset.shuffle(seed=42).select(100)`, per-question demo hash) into compact folders $HF_HOME/lighteval/helmet/<DATA_TAG>/base/<L>/<name>/ (one row per prompt: id, question,
context, demos, answers (list), extra (JSON string of the template fields and metric inputs), n_llama2) with manifest.json (sources: tarball sha256, code commit, dataset
revisions), and with `--model` the per-tokenizer copies $HF_HOME/lighteval/helmet/<DATA_TAG>/tok/<key>/<L>/<name>/ (+ n_tokens, cut_tokens) that the tasks read
(key = content hash of the tokenizer files, as RULER; the run exports TOKENIZER_PATH before lighteval imports this module).
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from lighteval.metrics.chat_metrics import ChatSamples
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------------------------------------- pins
HELMET_REPO = "princeton-nlp/HELMET"
HELMET_REVISION = "dddb209d03e38f1f0faf76d6d05ef4ccf96240ee"       # HF commit of the dataset repo (main, 2026-07-14)
HELMET_TARBALL = "data.tar.gz"                                      # 11,271,916,108 bytes; the README's `wget .../resolve/main/data.tar.gz`
HELMET_TARBALL_SHA256 = "9d693981aa3c065b8b2ff82ddf946141cdc4ece4524f18bff6f3fbd2a86982d9"   # the LFS oid of the tarball
HELMET_CODE_COMMIT = "aeadacc6593e9a222a84b0660b186a4e23d2efda"    # github.com/princeton-nlp/HELMET main (2026-09-19): data.py, utils.py, eval.py, configs/*, scripts/eval_gpt4_*.py
HELMET_CODE_LICENSE = "MIT (HELMET repo LICENSE, Copyright (c) 2024 Princeton Natural Language Processing)"
LLAMA2_TOKENIZER_REPO = "NousResearch/Llama-2-7b-hf"                # ungated copy; tokenizer.model sha256 9e556afd... = meta-llama/Llama-2-7b-hf's (HELMET's truncate_llama2 tokenizer)
LLAMA2_TOKENIZER_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
DATA_TAG = "v1"                                                     # prepared-folder tag: bump when the builders change the rows

SEED = 42                                                           # HELMET: --seed 42
N_QUESTIONS = 100                                                   # HELMET: --max_test_samples 100 (questions for RAG / re-rank; rows for json_kv, RULER, ICL)
N_ICL = 100                                                         # HELMET HEAD uses 500 since 2025-02-25; the paper and OLMES 100 (cost: docs/evals/tasks/helmet.md)
SHOTS = 2                                                           # HELMET: --shots 2 (demos of RAG / json_kv / re-rank / long documents); ICL has its own shots in the dataset name

LENGTHS = (8192, 16384, 32768, 65536, 131072)
CHAT_GEN = 4096                                                     # chat variant, NON-thinking: generation budget (the registry sets gen.max_new_tokens and max_model_len = L + CHAT_GEN; HELMET's own budgets are 20-200).
                                                                    # A THINKING run gets the chat budget (32,768, capped by the model's context) on top of the prompt window L: titan_eval config.resolve, task yaml `prompt_tokens: L`
CHAT_MARGIN = 64                                                    # tokens the chat template adds around the user message (prepare-time check)

# ----------------------------------------------------------------------------------------------------------- the subsets


@dataclass(frozen=True)
class Spec:
    name: str            # short name: helmet_<L>:<name>
    category: str        # Recall | RAG | Re-rank | ICL | LongQA | Summ
    kind: str            # json_kv | ruler | qa | rerank | icl | infbench_qa | infbench_choice | infbench_sum | narrativeqa | multi_lexsum
    gen: int             # HELMET generation_max_length (base variant)
    stop_nl: bool        # HELMET stop_new_line (base variant)
    judged: bool = False # scored by a judge (chat only)
    file: str = ""       # source folder / file stem inside the tarball


SPECS: dict[str, Spec] = {s.name: s for s in [
    Spec("json_kv", "Recall", "json_kv", 100, False),
    Spec("niah_mk_2", "Recall", "ruler", 50, False, file="niah_multikey_2"),
    Spec("niah_mk_3", "Recall", "ruler", 100, False, file="niah_multikey_3"),
    Spec("niah_mv", "Recall", "ruler", 50, False, file="niah_multivalue"),
    Spec("kilt_nq", "RAG", "qa", 20, True, file="nq-dev-multikilt_1000_k{k}_dep6"),
    Spec("kilt_triviaqa", "RAG", "qa", 20, True, file="triviaqa-dev-multikilt_1000_k{k}_dep6"),
    Spec("kilt_hotpotqa", "RAG", "qa", 20, True, file="hotpotqa-dev-multikilt_1000_k{k}_dep3"),
    Spec("kilt_popqa_3", "RAG", "qa", 20, True, file="popqa_test_1000_k{k}_dep6"),
    Spec("msmarco_rerank_psg", "Re-rank", "rerank", 200, True),
    Spec("icl_trec_coarse", "ICL", "icl", 20, True),
    Spec("icl_trec_fine", "ICL", "icl", 20, True),
    Spec("icl_banking77", "ICL", "icl", 20, True),
    Spec("icl_clinic150", "ICL", "icl", 20, True),
    Spec("icl_nlu", "ICL", "icl", 20, True),
    Spec("narrativeqa", "LongQA", "narrativeqa", 100, False, judged=True),
    Spec("infbench_qa_eng", "LongQA", "infbench_qa", 10, False),
    Spec("infbench_choice_eng", "LongQA", "infbench_choice", 10, False),
    Spec("infbench_sum_eng", "Summ", "infbench_sum", 1200, False, judged=True),
    Spec("multi_lexsum", "Summ", "multi_lexsum", 400, False, judged=True),
]}
CATEGORIES = ("Recall", "RAG", "Re-rank", "ICL", "LongQA", "Summ")
STAGE1 = tuple(n for n, s in SPECS.items() if s.category in ("Recall", "RAG", "Re-rank", "ICL"))
RULE_SCORED = STAGE1 + ("infbench_qa_eng", "infbench_choice_eng")          # scored by HELMET's own rule functions: base and chat variant
JUDGE_SCORED = ("narrativeqa", "infbench_sum_eng", "multi_lexsum")          # HELMET scores these with a GPT-4o judge: chat variant only, through the local judge stage

# per-length file selectors of the shipped data (HELMET configs/*_short.yaml, 131072 from configs/*.yaml)
JSON_KV_K = {8192: 105, 16384: 220, 32768: 440, 65536: 900, 131072: 1800}          # data/json_kv/test_k<K>_dep6.jsonl
KILT_K = {8192: 50, 16384: 105, 32768: 220, 65536: 440, 131072: 1000}              # data/kilt/<set>_k<K>_dep<D>.jsonl
MSMARCO_K = {8192: 50, 16384: 130, 32768: 285, 65536: 600, 131072: 1000}           # data/msmarco/test_reranking_data_k<K>_dep3.jsonl
ICL_SHOTS = {                                                                     # the <N>shot of the HELMET dataset names (icl_<set>_<N>shot_balance__<L>)
    "icl_trec_coarse": {8192: 400, 16384: 800, 32768: 1600, 65536: 3300, 131072: 6600},
    "icl_trec_fine": {8192: 400, 16384: 800, 32768: 1600, 65536: 3200, 131072: 6400},
    "icl_banking77": {8192: 360, 16384: 720, 32768: 1450, 65536: 2900, 131072: 5900},
    "icl_clinic150": {8192: 440, 16384: 880, 32768: 1750, 65536: 3525, 131072: 7050},
    "icl_nlu": {8192: 510, 16384: 1020, 32768: 2040, 65536: 4080, 131072: 8296},
}
# long documents: the number in HELMET's dataset name = the Llama-2 token budget of the CONTEXT (instruction and demos come on top)
DOC_TOKENS = {
    "narrativeqa": {8192: 7892, 16384: 16084, 32768: 32468, 65536: 65236, 131072: 130772},
    "infbench_qa_eng": {8192: 7982, 16384: 16174, 32768: 32558, 65536: 65326, 131072: 130862},
    "infbench_choice_eng": {8192: 7982, 16384: 16174, 32768: 32558, 65536: 65326, 131072: 130862},
    "infbench_sum_eng": {8192: 6792, 16384: 14984, 32768: 31368, 65536: 64136, 131072: 129672},
    "multi_lexsum": {8192: 7492, 16384: 15684, 32768: 32068, 65536: 64836, 131072: 130372},
}

# --------------------------------------------------------------------------------------------- HELMET prompt templates
# Verbatim from HELMET data.py @ HELMET_CODE_COMMIT. `{demo}` of the long-document templates is named `{demos}` here (same text, one field name for all kinds).
QA_USER = ("Use the given documents to write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\n"
           "{demos}{context}\n\nQuestion: {question}")
JSON_KV_USER = "{context}\n\nExtract the value corresponding to the specified key in the JSON object below.\n\n{demos}Key: {question}"
RERANK_USER = ("You are provided with a list of documents, each indicated by their ID. Rank each document based on their relevance to the question in descending order "
               "from most relelvant to least relevant texts. Include all documents in the rankings. Write your answer using the unique IDs, with the following format:\n"
               "Ranking: ID3 > ID1 > ID2\n\n{demos}{context}\n\nQuery: {question}")
ICL_USER = "Use the provided mapping from the text to label to assign a label to the text. Only output \"label: {{label}}\" and nothing else. \n\n{context}\n\n{question}"
RULER_USER_SINGLE = ("A special magic {type_needle_v} is hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
                     "{context}\nWhat is the special magic {type_needle_v} for {query} mentioned in the provided text?")
RULER_SYS_SINGLE = "The special magic {type_needle_v} for {query} mentioned in the provided text is"
RULER_USER_MULTI = ("Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
                    "{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?")
RULER_SYS_MULTI = "The special magic {type_needle_v} for {query} mentioned in the provided text are"

INFBENCH_QA_USER = ("You are given a story and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n"
                    "{demos}{context}\n\nQuestion: {question}")
INFBENCH_CHOICE_USER = ("You are given a story and a question with multiple choices. Choose the best answer from the options provided. Only one of the following options is correct, "
                        "output the answer using one single letter (A, B, C, or D). Don't say anything else.\n\n{demos}{context}\n\nQuestion: {question}\nOptions:\n{options}")
INFBENCH_SUM_USER = ("You are given a book and you are tasked to summarize it. Write a summary of about 1000 to 1200 words. Only write about the plot and characters of the story. "
                     "Do not discuss the themes or background of the book. Do not provide any analysis or commentary.\n\n{demos}{context}\n\nNow summarize the book.")
NARRATIVEQA_USER = ("You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n"
                    "{demos}{context}\n\nQuestion: {question}")
MULTI_LEXSUM_USER = ("You are given the legal documents in a civil rights lawsuit, and you are tasked to summarize the case. Write a concise summary of one paragraph (200 to 250 words). "
                     "The summary should contain a short description of the background, the parties involved, and the outcomes of the case.\n\n{demos}Legal documents:\n{context}\n\nNow please summarize the case.")

# kind -> (user_template, system_template (the answer prefix), separator between them in the base prompt)
TEMPLATES: dict[str, tuple[str, str, str]] = {
    "infbench_qa": (INFBENCH_QA_USER, "Answer:", "\n\n"),
    "infbench_choice": (INFBENCH_CHOICE_USER, "Answer:", "\n\n"),
    "infbench_sum": (INFBENCH_SUM_USER, "Summary:", "\n\n"),
    "narrativeqa": (NARRATIVEQA_USER, "Answer:", "\n"),
    "multi_lexsum": (MULTI_LEXSUM_USER, "Summary:", "\n\n"),
    "qa": (QA_USER, "Answer:", "\n"),
    "json_kv": (JSON_KV_USER, "Corresponding value:", "\n"),
    "rerank": (RERANK_USER, "Ranking:", "\n"),
    "icl": (ICL_USER, "label:", "\n"),
}
RULER_TEMPLATES = {"niah_mk_2": (RULER_USER_SINGLE, RULER_SYS_SINGLE), "niah_mk_3": (RULER_USER_SINGLE, RULER_SYS_SINGLE), "niah_mv": (RULER_USER_MULTI, RULER_SYS_MULTI)}


def templates_of(spec: Spec) -> tuple[str, str, str]:
    """(user_template, system_template, separator) of a subset."""
    if spec.kind == "ruler":
        u, s = RULER_TEMPLATES[spec.name]
        return u, s, "\n"
    return TEMPLATES[spec.kind]


# ----------------------------------------------------------------------------------------- HELMET's metric functions (utils.py)
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    np_, ng = normalize_answer(prediction), normalize_answer(ground_truth)
    if np_ in ["yes", "no", "noanswer"] and np_ != ng:
        return 0.0
    if ng in ["yes", "no", "noanswer"] and np_ != ng:
        return 0.0
    pt, gt = np_.split(), ng.split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def _max_over(fn, prediction: str, answers) -> float:
    if isinstance(answers, str):
        answers = [answers]
    elif answers and isinstance(answers[0], list):
        answers = [g for gl in answers for g in gl]
    return max(fn(prediction, a) for a in answers)


_ROUGE = None


def _rouge():
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer

        _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE


def calculate_metrics(prediction: str, answers) -> dict[str, float]:
    """HELMET utils.calculate_metrics (rougeLsum left out: no category uses it)."""
    em = _max_over(exact_match_score, prediction, answers)
    f1 = _max_over(f1_score, prediction, answers)
    sub = _max_over(substring_exact_match_score, prediction, answers)
    if isinstance(answers, str):
        answers = [answers]
    elif answers and isinstance(answers[0], list):
        answers = [g for gl in answers for g in gl]
    rl = [_rouge().score(target=a, prediction=prediction)["rougeL"] for a in answers]
    return {"exact_match": float(em), "f1": float(f1), "substring_exact_match": float(sub),
            "rougeL_f1": max(r.fmeasure for r in rl), "rougeL_recall": max(r.recall for r in rl)}


def parse_output(output: str, prefix: str = "Answer:"):
    """HELMET utils.parse_output: the rest of the line after the first `prefix` (case-insensitive), else the first line; the prefix is stripped again."""
    def lstrip_string(s, sub):
        return re.sub(f"^{re.escape(sub)}", "", s, flags=re.IGNORECASE)

    patterns = [re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), re.compile(r"(?:^)(.*)(?:\n|$)")]
    for pat in patterns:
        matches = pat.search(output)
        if matches is not None:
            return lstrip_string(matches[1].strip(), prefix).strip()
    return None


def parse_rankings(output: str) -> dict[str, int]:
    """HELMET utils.parse_rankings: {id: score} with score = len - position of the longest `ID > ID > ...` chain of the output."""
    output = re.sub(r"[\[\]:]", "", output)
    output = output.lower().replace("id", "")
    pattern = r"(\d+)(?:\s*>\s*(\d+))*"
    longest = ""
    for m in re.finditer(pattern, output):
        if len(m.group(0)) > len(longest):
            longest = m.group(0)
    if len(longest) > 0:
        rankings = [num.strip() for num in longest.split(">") if num.strip().isdigit()]
    else:
        rankings = [output]
    results: dict[str, int] = {}
    for i, rank in enumerate(rankings):
        if rank not in results:
            results[rank] = len(rankings) - i
    return results


def ndcg_cut(run: dict[str, float], qrels: dict[str, int], k: int) -> float:
    """trec_eval ndcg_cut.<k> (what pytrec_eval, hence HELMET's NDCG@k, computes; pytrec_eval is not installed): gain = the relevance label,
    discount 1 / log2(rank + 1), documents ranked by run score (ties: larger doc id first, as trec_eval), IDCG over all judged documents."""
    ranked = sorted(run.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:k]
    dcg = sum(qrels.get(doc, 0) / math.log2(i + 2) for i, (doc, _) in enumerate(ranked))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------------------------------- scoring one completion
def prefixed(spec: Spec, fields: dict, completion: str, chat: bool) -> str:
    """HELMET eval.py: a completion (no chat template) is scored together with the answer prefix that ended the prompt. A chat reply is scored as it is, except for the
    datasets with `stop_new_line` (RAG, re-rank, ICL): HELMET stops their generation at the first newline whatever the template, so the reply counts up to its first line
    (leading blank lines skipped; the generation budget of the chat variant is 4096 tokens non-thinking and up to 32,768 thinking, not HELMET's 20 / 200: without this a long reply that mentions the gold would score)."""
    if chat:
        return completion.lstrip().split("\n")[0] if spec.stop_nl else completion
    return templates_of(spec)[1].format(**fields) + completion


def score_completion(spec: Spec, fields: dict, answers: list[str], completion: str, chat: bool) -> dict[str, float]:
    """The HELMET post_process of the subset's kind. `fields`: the template fields of the row (question, query, type_needle_v, ... and `qrel`)."""
    out_text = prefixed(spec, fields, completion, chat)
    kind = spec.kind
    if kind == "json_kv":
        m = calculate_metrics(out_text, answers)
        parsed = parse_output(out_text, "corresponding value:")
        m2 = calculate_metrics(parsed, answers)
        m = {k: max(v, m2[k]) for k, v in m.items()}
        return {"helmet_score": m["substring_exact_match"], "exact_match": m["exact_match"], "f1": m["f1"]}
    if kind == "ruler":
        recall = sum(a.lower() in out_text.lower() for a in answers) / len(answers)
        return {"helmet_score": recall}
    if kind == "qa":
        m = calculate_metrics(out_text, answers)
        parsed = parse_output(out_text)
        if parsed is not None:
            m2 = calculate_metrics(parsed, answers)
            m = {k: max(v, m2[k]) for k, v in m.items()}
        return {"helmet_score": m["substring_exact_match"], "exact_match": m["exact_match"], "f1": m["f1"]}
    if kind == "icl":
        pred = parse_output(out_text, "label:")
        m = calculate_metrics(pred, answers)
        return {"helmet_score": m["exact_match"], "f1": m["f1"]}
    if kind == "rerank":
        run = parse_rankings(out_text)
        qrels = {c[0]: int(c[1]) for c in fields["qrel"]}
        vals = {k: ndcg_cut(run, qrels, k) for k in (1, 5, 10)}
        return {"helmet_score": vals[10], "ndcg@1": vals[1], "ndcg@5": vals[5]}
    return score_stage2(spec, fields, answers, out_text)


def score_stage2(spec: Spec, fields: dict, answers: list[str], out_text: str) -> dict[str, float]:
    """Long-document kinds scored by a rule: infbench_qa (ROUGE-L F1 = helmet_score), infbench_choice (exact match, HELMET's choice_post_process).
    narrativeqa / infbench_sum / multi_lexsum have no rule score: the judge stage computes their helmet_score (chat variant only)."""
    if spec.kind == "infbench_qa":
        m = calculate_metrics(out_text, answers)
        parsed = parse_output(out_text)
        if parsed is not None:
            m2 = calculate_metrics(parsed, answers)
            m = {k: max(v, m2[k]) for k, v in m.items()}
        return {"helmet_score": m["rougeL_f1"], "exact_match": m["exact_match"], "f1": m["f1"]}
    if spec.kind == "infbench_choice":
        m = calculate_metrics(out_text, answers)                      # answers = [letter, "letter. option text"]
        m.pop("substring_exact_match")
        parsed = parse_output(out_text)
        if parsed is not None:
            m2 = calculate_metrics(parsed, answers)
            m2.pop("substring_exact_match")
            m = {k: max(v, m2[k]) for k, v in m.items()}
        em = m["exact_match"]
        full = answers[1].lower() in out_text.lower()                 # HELMET: the full option text anywhere in the output also counts (merged into exact_match)
        if full:
            em = 1.0
        # HELMET's exact match normalises with `remove_articles`, which turns the letter "A" into "": an empty / punctuation-only answer is "exactly" the gold A (about 20% of
        # the items). `acc_strict` compares the letter itself (parsed text, brackets and dots stripped) and keeps the full-option rule; it is a diagnostic, helmet_score stays HELMET's.
        cand = (parsed or "").strip().strip(".:)(*[]\"' ").upper()
        strict = 1.0 if (full or (len(cand) == 1 and cand == answers[0])) else 0.0
        return {"helmet_score": em, "acc_strict": strict}
    raise ValueError(f"{spec.name}: no rule score (judge-graded)")


# lighteval pads a metric that some subtasks of a parent task lack with 0.0 in every other subtask (MetricsLogger.aggregate reads the first subtask's names from all of them),
# so every subtask reports the same single metric. `score_completion` also returns diagnostics (exact_match, f1, ndcg@1 / @5, acc_strict): the unit tests and
# agents/reward_checks/helmet_checks.py read them, the lighteval metric does not carry them.
METRIC_NAMES = {kind: ("helmet_score",) for kind in ("json_kv", "ruler", "qa", "rerank", "icl", "infbench_qa", "infbench_choice")}


# ----------------------------------------------------------------------------------------------------------- prompts
def _fields(line: dict) -> dict:
    f = {"question": line.get("question", ""), "context": line["context"], "demos": line.get("demos", "")}
    f.update(json.loads(line["extra"]) if line.get("extra") else {})
    return f


def build_prompt(spec: Spec, line: dict, chat: bool) -> str:
    user, system, sep = templates_of(spec)
    fields = _fields(line)
    prompt = user.format(**fields)
    return prompt if chat else prompt + sep + system.format(**fields)


def scoring_fields(line: dict) -> dict:
    """What the scorers need besides the answers: the small template fields (question, query, type_needle_v, qrel, ...), NOT the context / demos (the prompt is stored with the
    response anyway; a second copy of 100k tokens per doc would double the details files)."""
    f = {"question": line.get("question", "")}
    f.update(json.loads(line["extra"]) if line.get("extra") else {})
    return f


def make_prompt_function(spec: Spec, chat: bool) -> Callable:
    def prompt(line, task_name: str = None):
        fields = scoring_fields(line)
        return Doc(
            task_name=task_name,
            query=build_prompt(spec, line, chat),
            choices=[""],                       # ChatSamples needs one gold; the scorer reads `specific`
            gold_index=0,
            specific={"fields": fields, "answers": list(line["answers"]), "id": line.get("id")},
        )

    prompt.__name__ = f"helmet_{spec.name}_{'chat' if chat else 'base'}_prompt"
    return prompt


def make_score_function(spec: Spec, chat: bool) -> Callable[[Doc, str], dict[str, float]]:
    def score(doc: Doc, pred: str) -> dict[str, float]:
        sp = doc.specific
        return score_completion(spec, sp["fields"], sp["answers"], pred, chat)

    return score


def make_metric(spec: Spec, chat: bool) -> SampleLevelMetricGrouping:
    names = METRIC_NAMES[spec.kind]
    return SampleLevelMetricGrouping(
        metric_name=list(names),
        sample_level_fn=ChatSamples(1, make_score_function(spec, chat), names, ks=()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={n: np.mean for n in names},
        higher_is_better={n: True for n in names},
    )


# ----------------------------------------------------------------------------------------------------------- task table
def helmet_root() -> Path:
    return Path(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "lighteval" / "helmet" / DATA_TAG


def base_dir(length: int, name: str) -> Path:
    """The tokenizer-independent prepared rows (HELMET's Llama-2 lengths)."""
    return helmet_root() / "base" / str(length) / name


def tokenizer_key(tokenizer_path: str) -> str:
    from lighteval.tasks.tasks.ruler import _get_tokenizer_hash          # one key scheme for every per-tokenizer cache (content hash of the tokenizer files)

    return _get_tokenizer_hash(tokenizer_path)


def tok_root(key: str) -> Path:
    return helmet_root() / "tok" / key


def tok_dir(key: str, length: int, name: str) -> Path:
    return tok_root(key) / str(length) / name


def task_name(length: int, name: str, chat: bool) -> str:
    return f"helmet_{length}{'_chat' if chat else ''}:{name}"


def get_helmet_tasks(tokenizer_path: str | None, names: tuple[str, ...] | None = None, lengths: tuple[int, ...] = LENGTHS) -> list[LightevalTaskConfig]:
    """Task configs of both variants. The rows come from the per-tokenizer folder of `tokenizer_path` (TOKENIZER_PATH of the run); without a tokenizer the path
    is a placeholder so that the names can be listed (plan, registry checks) and nothing can be run."""
    key = tokenizer_key(tokenizer_path) if tokenizer_path else "unset"
    tasks = []
    for chat in (False, True):
        for length in lengths:
            for name in names or AVAILABLE:
                spec = SPECS[name]
                if chat is False and spec.judged:
                    continue                     # judge-only subsets: chat variant only
                tasks.append(LightevalTaskConfig(
                    name=task_name(length, name, chat),
                    prompt_function=make_prompt_function(spec, chat),
                    hf_repo=str(tok_dir(key, length, name)),         # local path -> load_from_disk
                    hf_subset="default",
                    hf_avail_splits=["test"],
                    evaluation_splits=["test"],
                    few_shots_split=None,
                    few_shots_select=None,
                    generation_size=CHAT_GEN if chat else spec.gen,
                    stop_sequence=[] if chat else (["\n"] if spec.stop_nl else None),
                    metrics=[make_metric(spec, chat)],
                    version=0,
                ))
    return tasks


# =============================================================================================================== prepare
# Login node only (`titan-eval prepare helmet`): reads the HELMET tarball (HF cache) and the ICL sources, writes the prepared folders. Nothing here runs in a job.
SRC_NLU_COMMIT = "f6071b496b17d71e6eb43f543af0707f4ff30557"        # github.com/xliuhw/NLU-Evaluation-Data (CC BY 4.0), the HF repo xingkunliuxtracta/nlu_evaluation_data no longer exists
SRC_NLU_FILE = "AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv"
SRC_NLU_SHA256 = "5f6dbf6d38fc111217924945ac59c554e0b926d5aa836ecdd0d089d2ca48e1d9"
SRC_NLU_URL = f"https://raw.githubusercontent.com/xliuhw/NLU-Evaluation-Data/{SRC_NLU_COMMIT}/{SRC_NLU_FILE}"
ICL_SOURCES = {                                                    # hub-cache folders (login-node `snapshot_download`, pinned in configs/datasets.lock.yaml)
    "trec": ("CogComp/trec", "65752bf53af25bc935a0dce92fb5b6c930728450", "refs/convert/parquet of the script-only repo: default/{train,test}/0000.parquet"),
    "banking77": ("PolyAI/banking77", "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6", "data/{train,test}-00000-of-00001.parquet"),
    "clinc": ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec", "plus/{train,validation}-00000-of-00001.parquet"),
}
ICL_LABELS = {"icl_trec_coarse": 6, "icl_trec_fine": 50, "icl_banking77": 77, "icl_clinic150": 151, "icl_nlu": 68}


def _hf_home() -> Path:
    return Path(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))


def src_root() -> Path:
    """Where the needed members of the tarball are extracted (login node, deletable after the build: `prepare helmet --clean`)."""
    return _hf_home() / "lighteval" / "helmet_src" / HELMET_REVISION[:8]


def tarball_path() -> Path:
    p = _hf_home() / "hub" / "datasets--princeton-nlp--HELMET" / "snapshots" / HELMET_REVISION / HELMET_TARBALL
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: on the login node run `hf_hub_download('{HELMET_REPO}', '{HELMET_TARBALL}', repo_type='dataset', revision='{HELMET_REVISION}')`")
    if p.resolve().name != HELMET_TARBALL_SHA256:
        raise ValueError(f"{p} is not the pinned tarball (blob {p.resolve().name}, expected {HELMET_TARBALL_SHA256})")
    return p


def hf_permutation(n: int, seed: int) -> np.ndarray:
    """What `datasets.Dataset.shuffle(seed=seed)` does to the row order: np.random.default_rng(seed).permutation(n) (checked against datasets in the unit test)."""
    return np.random.default_rng(seed).permutation(n)


def member_files(lengths=LENGTHS, names=None) -> list[str]:
    """Tarball members the given subsets need (paths inside data.tar.gz)."""
    out: set[str] = set()
    for name in names or STAGE1 + ("infbench_sum_eng", "multi_lexsum"):
        spec = SPECS[name]
        for L in lengths:
            if spec.kind == "json_kv":
                out.add(f"data/json_kv/test_k{JSON_KV_K[L]}_dep6.jsonl")
            elif spec.kind == "ruler":
                out.add(f"data/ruler/{spec.file}/validation_{L}.jsonl")
            elif spec.kind == "qa":
                out.add(f"data/kilt/{spec.file.format(k=KILT_K[L])}.jsonl")
                out.add("data/kilt/" + _demo_file(spec) + ".jsonl")
            elif spec.kind == "rerank":
                out.add(f"data/msmarco/test_reranking_data_k{MSMARCO_K[L]}_dep3.jsonl")
                out.add("data/msmarco/test_reranking_data_k10_dep3.jsonl")
        if spec.kind == "infbench_sum":
            out.add("data/infbench/longbook_sum_eng_keypoints.jsonl")
        if spec.kind == "multi_lexsum":
            out.add("data/multi_lexsum/multi_lexsum_val.jsonl")
    return sorted(out)


def _demo_file(spec: Spec) -> str:
    return {"kilt_nq": "nq-train-multikilt_1000_k3_dep6", "kilt_triviaqa": "triviaqa-train-multikilt_1000_k3_dep6",
            "kilt_hotpotqa": "hotpotqa-train-multikilt_1000_k3_dep3", "kilt_popqa_3": "popqa_test_1000_k3_dep6"}[spec.name]


def extract(members: list[str]) -> list[str]:
    """Extract the members that are not in src_root() yet (one `tar` pass over the 11 GB tarball, ~3 min). Returns the members it extracted."""
    import subprocess
    import tempfile

    root = src_root()
    missing = [m for m in members if not (root / m).is_file() or (root / m).stat().st_size == 0]
    if not missing:
        return []
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(missing) + "\n")
        lst = f.name
    try:
        subprocess.run(["tar", "xzf", str(tarball_path()), "-C", str(root), "-T", lst], check=True)
    finally:
        os.unlink(lst)
    still = [m for m in missing if not (root / m).is_file()]
    if still:
        raise FileNotFoundError(f"not in the tarball: {still}")
    return missing


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _read_jsonl(path: Path) -> list[dict]:
    return list(_iter_jsonl(path))


def _row_id(*parts) -> str:
    return "|".join(str(p) for p in parts)


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _first_unique(perm, items, key: str, n: int) -> list[dict]:
    """HELMET: `demos.shuffle(seed=h)`, `drop_duplicates(demos, key)`, `.select(range(shots))`: the first n rows of the permuted list with distinct `key`."""
    seen, out = set(), []
    for i in perm:
        d = items[i]
        if d[key] in seen:
            continue
        seen.add(d[key])
        out.append(d)
        if len(out) == n:
            break
    return out


def _sha_seed(value, abs_: bool = False) -> int:
    h = int(hashlib.sha256(str(value).encode("utf-8")).hexdigest(), 16) % 2**31
    return abs(h) if abs_ else h


def build_json_kv(spec: Spec, length: int) -> list[dict]:
    rows = _read_jsonl(src_root() / f"data/json_kv/test_k{JSON_KV_K[length]}_dep6.jsonl")
    demo_template = "Key: {key}\nCorresponding value:{value}"
    for r in rows:
        r["demos"] = "\n\n".join(demo_template.format(key=k, value=" " + v) for k, v in r["demos"][:SHOTS]) + ("\n\n" if SHOTS > 0 else "")
    sel = hf_permutation(len(rows), SEED)[: min(N_QUESTIONS, len(rows))]
    out = []
    for j, i in enumerate(sel):
        r = rows[int(i)]
        out.append({"id": _row_id(r["question"], r["depth"], j), "question": r["question"], "context": r["context"], "demos": r["demos"], "answers": [r["answer"]],
                    "extra": _dump({"depth": r["depth"], "k": r["num_kvs"]})})
    return out


def build_ruler(spec: Spec, length: int) -> list[dict]:
    rows = _read_jsonl(src_root() / f"data/ruler/{spec.file}/validation_{length}.jsonl")
    sel = hf_permutation(len(rows), SEED)[: min(N_QUESTIONS, len(rows))]
    out = []
    for i in sel:
        r = rows[int(i)]
        out.append({"id": _row_id("ruler", r["index"]), "question": r["query"], "context": r["context"], "demos": "", "answers": list(r["answer"]),
                    "extra": _dump({"type_needle_v": r["type_needle_v"], "query": r["query"], "ruler_length": r["length"]})})
    return out


def build_qa(spec: Spec, length: int) -> list[dict]:
    path = src_root() / "data/kilt" / (spec.file.format(k=KILT_K[length]) + ".jsonl")
    demo_rows = _read_jsonl(src_root() / "data/kilt" / (_demo_file(spec) + ".jsonl"))
    popqa = "popqa" in spec.name
    threshold = float(spec.name.split("_")[-1]) if popqa else None

    def keep(r):
        return not popqa or math.log10(r["s_pop"]) < threshold

    demo_rows = [d for d in demo_rows if keep(d)]
    # pass 1: the question keys (HELMET: key = "id" when the file has one, else "question"; keys = set(data[key]); random.seed(42); random.sample(sorted(keys), 100));
    # the rows of a chosen key are ALL kept (every depth)
    keys, offsets = [], []
    with open(path, "rb") as f:
        key = "id" if "id" in json.loads(f.readline()) else "question"
        f.seek(0)
        pos = f.tell()
        for line in iter(f.readline, b""):
            if line.strip():
                r = json.loads(line)
                if keep(r):
                    keys.append(r[key])
                    offsets.append(pos)
            pos = f.tell()
    chosen = set(random.Random(SEED).sample(sorted(set(keys)), min(N_QUESTIONS, len(set(keys)))))
    demo_template = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"
    passage_template = "Document (Title: {title}): {text}"
    out = []
    seen_ids: Counter = Counter()
    with open(path, "rb") as f:
        for k, off in zip(keys, offsets):
            if k not in chosen:
                continue
            f.seek(off)
            r = json.loads(f.readline())
            pool = [d for d in demo_rows if d[key] != r[key]] if popqa else demo_rows
            picked = _first_unique(hf_permutation(len(pool), _sha_seed(r[key])), pool, key, SHOTS)
            demos = "\n\n".join(demo_template.format(documents="\n\n".join(passage_template.format(**c) for c in d["ctxs"]), question=d["question"], answer=d["answers"][0])
                                for d in picked) + "\n\n"
            context = "\n\n".join(passage_template.format(**c) for c in r["ctxs"]) if len(r["ctxs"]) > 0 else ""
            rid = _row_id(r[key], r.get("depth"))
            seen_ids[rid] += 1                          # HotpotQA's dep3 rows have no depth field: the n-th copy of a question
            out.append({"id": f"{rid}|{seen_ids[rid]}", "question": r["question"], "context": context, "demos": demos, "answers": list(r["answers"]),
                        "extra": _dump({"depth": r.get("depth"), "k": len(r["ctxs"])})})
    return out


def build_rerank(spec: Spec, length: int) -> list[dict]:
    data = _read_jsonl(src_root() / f"data/msmarco/test_reranking_data_k{MSMARCO_K[length]}_dep3.jsonl")
    demos_all = _read_jsonl(src_root() / "data/msmarco/test_reranking_data_k10_dep3.jsonl")
    chosen = set(random.Random(SEED).sample(sorted({r["qid"] for r in data}), min(N_QUESTIONS, len({r["qid"] for r in data}))))
    out = []
    seen_ids: Counter = Counter()
    for r in data:
        if r["qid"] not in chosen:
            continue
        passage_template = "[ID: {id}] Document (Title: {title}): {text}" if "title" in r["ctxs"][0] else "[ID: {id}] Document: {text}"
        context = "\n\n".join(passage_template.format(**c) for c in r["ctxs"])
        gold = " > ".join(x["id"] for x in sorted(r["ctxs"], key=lambda x: x["label"], reverse=True))
        pool = demos_all if len(demos_all) > 2 * len(data) else [d for d in demos_all if d["qid"] != r["qid"]]
        picked = _first_unique(hf_permutation(len(pool), _sha_seed(r["qid"], True)), pool, "qid", SHOTS)
        demo_text = ""
        for d in picked:
            ranking = " > ".join(x["id"] for x in sorted(d["ctxs"], key=lambda x: x["label"], reverse=True))
            demo_text += "\n\n".join(passage_template.format(**c) for c in d["ctxs"]) + f"\n\nQuery: {d['query']}\nRanking: {ranking}" + "\n\n"
        seen_ids[r["qid"]] += 1                                        # dep3: the same query three times (the relevant passage at three positions)
        out.append({"id": _row_id("msmarco", r["qid"], seen_ids[r["qid"]]), "question": r["query"], "context": context, "demos": demo_text, "answers": [gold],
                    "extra": _dump({"qid": r["qid"], "qrel": [[c["id"], str(c["label"])] for c in r["ctxs"]], "k": len(r["ctxs"])})})
    return out


def _balance_labels(data: list[dict], shots: int, seed: int, label_field: str) -> list[dict]:
    """HELMET data.py balance_labels, verbatim."""
    rand = random.Random(seed)
    label_mapping: dict = {x[label_field]: [] for x in data}
    for x in data:
        label_mapping[x[label_field]].append(x)
    num_rounds = math.ceil(shots / len(label_mapping))
    new_data: list[list] = [[] for _ in range(num_rounds)]
    for _, samples in label_mapping.items():
        indices = rand.sample(range(len(samples)), num_rounds % len(samples))
        while len(indices) < num_rounds:
            indices += rand.sample(range(len(samples)), min(num_rounds - len(indices), len(samples)))
        for i, idx in enumerate(indices):
            new_data[i].append(samples[idx])
    for i in range(len(new_data)):
        rand.shuffle(new_data[i])
    return [item for sublist in new_data for item in sublist][:shots]


def _read_parquet_rows(folder_glob: str) -> list[dict]:
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(folder_glob))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one parquet file for {folder_glob}, found {files}")
    return pq.read_table(files[0]).to_pylist()


def icl_source(name: str) -> tuple[list[dict], list[dict], str, str]:
    """(train rows, test rows, text field, label field) of an ICL dataset exactly as HELMET loads it (script loaders replaced by the pinned parquet files)."""
    hub = _hf_home() / "hub"

    def snap(key):
        repo, rev, _ = ICL_SOURCES[key]
        return hub / ("datasets--" + repo.replace("/", "--")) / "snapshots" / rev

    if name in ("icl_trec_coarse", "icl_trec_fine"):
        s = snap("trec")
        label = "coarse_label" if name.endswith("coarse") else "fine_label"
        return _read_parquet_rows(f"{s}/default/train/0000.parquet"), _read_parquet_rows(f"{s}/default/test/0000.parquet"), "text", label
    if name == "icl_banking77":
        s = snap("banking77")
        return (_read_parquet_rows(f"{s}/data/train-00000-of-00001.parquet"), _read_parquet_rows(f"{s}/data/test-00000-of-00001.parquet"), "text", "label")
    if name == "icl_clinic150":
        s = snap("clinc")
        return (_read_parquet_rows(f"{s}/plus/train-00000-of-00001.parquet"), _read_parquet_rows(f"{s}/plus/validation-00000-of-00001.parquet"), "text", "intent")
    if name == "icl_nlu":
        rows = nlu_rows()
        perm = hf_permutation(len(rows), SEED)           # Dataset.train_test_split(test_size=0.1, seed=42): shuffled, the first 10% are the test split
        n_test = math.ceil(0.1 * len(rows))
        test = [rows[int(i)] for i in perm[:n_test]]
        train = [rows[int(i)] for i in perm[n_test: n_test + (len(rows) - n_test)]]
        return train, test, "text", "label"
    raise KeyError(name)


def nlu_path() -> Path:
    return _hf_home() / "lighteval" / "helmet_src" / "nlu" / SRC_NLU_COMMIT[:8] / "NLU-Data-Home-Domain-Annotated-All.csv"


def nlu_rows() -> list[dict]:
    """NLU Evaluation Data (Liu et al. 2019) as the HF dataset had it: text = the normalised utterance (the entity-stripped annotation where it is missing),
    label = the id of scenario_intent among the 68 sorted names. Reconstructed from the official CSV (the HF repo is gone): the row order is the CSV's."""
    import csv

    p = nlu_path()
    if hashlib.sha256(p.read_bytes()).hexdigest() != SRC_NLU_SHA256:
        raise ValueError(f"{p}: sha256 differs from the pinned file {SRC_NLU_SHA256}")
    with open(p, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f, delimiter=";"))
    names = sorted({f"{r['scenario']}_{r['intent']}" for r in raw})
    out = []
    for r in raw:
        text = r["answer_normalised"]
        if text in ("", "null"):
            text = re.sub(r"\[[^\]:]*:\s*([^\]]*)\]", r"\1", r["answer_annotation"]).strip()
        if text in ("", "null"):
            continue
        out.append({"text": text, "label": names.index(f"{r['scenario']}_{r['intent']}")})
    return out


def build_icl(spec: Spec, length: int) -> list[dict]:
    """HELMET data.py load_icl (HEAD): N_ICL test items with balanced labels, each with its own `shot` demos (balanced over labels, seeded by the item's text hash)
    and its own random label-id mapping; the answer is the mapped id."""
    shot = ICL_SHOTS[spec.name][length]
    num_labels = ICL_LABELS[spec.name]
    train, test, text_field, label_field = icl_source(spec.name)
    if len(test) > N_ICL:
        test = [test[int(i)] for i in hf_permutation(len(test), SEED)]      # test_data.shuffle(seed=seed)
        test = _balance_labels(test, N_ICL, SEED, label_field)
    item_template = "{text}\nlabel: {label}"
    out = []
    for j, sample in enumerate(test):
        local_seed = (int(hashlib.sha256(sample[text_field].encode("utf-8")).hexdigest(), 16) + SEED) % 2**31
        demos = _balance_labels(train, shot, local_seed, label_field)
        mapping = list(range(num_labels))
        random.Random(local_seed).shuffle(mapping)
        context = "\n\n".join(item_template.format(text=d[text_field], label=str(mapping[int(d[label_field])])) for d in demos)
        out.append({"id": _row_id(spec.name, j), "question": sample[text_field], "context": context, "demos": "", "answers": [str(mapping[int(sample[label_field])])],
                    "extra": _dump({"shots": shot})})
    return out


# ---- long documents: HELMET `truncate_llama2` (the context is cut to the budget of the length in Llama-2 tokens, then " ... [the rest of the text is omitted]")
TRUNCATION_POSTFIX = " ... [the rest of the text is omitted]"
INFBENCH_SPLITS = {"infbench_qa": "longbook_qa_eng", "infbench_choice": "longbook_choice_eng", "infbench_sum": "longbook_sum_eng"}
SRC_INFBENCH = ("xinrongzhang2022/InfiniteBench", "90f0394333616266d9fe85824ceaf505093cbaa5", "Apache-2.0 (HF card)")
SRC_LEXSUM = ("allenai/multi_lexsum", "055f5fa5300a5fdefb5b691c4d6c1e58f485b379", "ODC-BY (HF card); v20230518 train.json for the demo summaries, the validation cases come with the HELMET tarball")
SRC_NQA = ("deepmind/narrativeqa", "2e643e7363944af1c33a652d1c87320d0871c4e4", "Apache-2.0 (HF card)")
MIN_DOC_TOKENS = {"infbench_qa": 65536, "infbench_choice": 65536, "infbench_sum": 65536, "multi_lexsum": 65536, "narrativeqa": 131072}   # HELMET filter_length thresholds


def _demo_seed(key: str, seed: int = SEED) -> int:
    """HELMET demo_seed: a seed per sample, deterministic and shifted by the run seed."""
    return (int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) + seed) % 2**31


def _hub_file(src: tuple, rel: str) -> Path:
    repo, rev, _ = src
    return _hf_home() / "hub" / ("datasets--" + repo.replace("/", "--")) / "snapshots" / rev / rel


def llama2_encode_many(texts: list[str], batch: int = 12) -> list[tuple[int, np.ndarray]]:
    """[(token count with <s>, end offset of every token)]: what `tokenizer(text, return_offsets_mapping=True)` gives HELMET's truncate_llama2. Llama-2's BPE runs over the whole
    text as one piece (slow for a 3 MB book), so the documents go through `encode_batch` (parallel over documents), a few at a time to bound the memory."""
    tok = _llama2_tokenizer()
    out = []
    for i in range(0, len(texts), batch):
        for e in tok.encode_batch(texts[i: i + batch], add_special_tokens=True):
            out.append((len(e.ids), np.asarray([o[1] for o in e.offsets], dtype=np.int32)))
    return out


def llama2_encode(text: str) -> tuple[int, np.ndarray]:
    return llama2_encode_many([text])[0]


def truncate_llama2(text: str, n_tokens: int, ends: np.ndarray, max_length: int) -> str:
    """HELMET data.py truncate_llama2: context[:offsets[max_length - separator_length][1]] + postfix when the context is longer than max_length Llama-2 tokens."""
    if n_tokens <= max_length:
        return text
    sep = len(_llama2_tokenizer().encode(TRUNCATION_POSTFIX, add_special_tokens=True).ids)
    return text[: int(ends[max_length - sep])] + TRUNCATION_POSTFIX


_DOC_CACHE: dict[str, list[dict]] = {}


def _infbench_docs(kind: str) -> list[dict]:
    """The sampled documents of an InfiniteBench kind, with their Llama-2 encodings (identical for every length): HELMET load_infbench (HEAD): the demos come from the other
    items of the whole split (`shuffle(seed=42)`, first 2), then filter_length(>= 65536 tokens), then shuffle(seed=42).select(100)."""
    if kind in _DOC_CACHE:
        return _DOC_CACHE[kind]
    rows = _read_jsonl(_hub_file(SRC_INFBENCH, INFBENCH_SPLITS[kind] + ".jsonl"))
    for r in rows:
        r["question"] = r["input"]
        if kind == "infbench_choice":
            r["options_text"] = "A. {}\nB. {}\nC. {}\nD. {}".format(*r["options"])
            letter = chr(ord("A") + r["options"].index(r["answer"][0]))
            r["answers"] = [letter, f"{letter}. {r['answer'][0]}"]
        else:
            r["answers"] = list(r["answer"])
    keyp = {}
    if kind == "infbench_sum":
        keyp = {d["id"]: d["keypoints"] for d in _iter_jsonl(src_root() / "data/infbench/longbook_sum_eng_keypoints.jsonl")}

    def demos_of(r):
        others = [x for x in rows if x["id"] != r["id"]]
        picked = [others[int(i)] for i in hf_permutation(len(others), SEED)[:SHOTS]]
        if kind == "infbench_qa":
            body = "\n\n".join(f"[story text]\nQuestion: {x['question']}\nAnswer: {x['answers'][0]}" for x in picked)
        elif kind == "infbench_choice":
            body = "\n\n".join(f"[story text]\nQuestion: {x['question']}\nOptions:\n{x['options_text']}\nAnswer: {x['answers'][0]}" for x in picked)
        else:
            body = "\n\n".join(f"[story text]\nSummary: {x['answers'][0].strip()}" for x in picked)
        return f"For example:\n\n{body}\n\nNow, read the following story:\n\n"

    for r in rows:
        r["demos"] = demos_of(r)
    enc = llama2_encode_many([r["context"] for r in rows])
    keep = [i for i, (n, _) in enumerate(enc) if n >= MIN_DOC_TOKENS[kind]]
    sel = [keep[int(i)] for i in hf_permutation(len(keep), SEED)[: min(N_QUESTIONS, len(keep))]]
    docs = []
    for i in sel:
        r = rows[i]
        extra = {"options": r.get("options_text"), "src_id": r["id"], "doc_llama2_tokens": enc[i][0]}
        if kind == "infbench_sum":
            extra["keypoints"] = keyp.get(r["id"])
        docs.append({"id": _row_id(kind, r["id"]), "question": r["question"], "context": r["context"], "demos": r["demos"], "answers": r["answers"], "extra": extra,
                     "_n": enc[i][0], "_ends": enc[i][1]})
    _DOC_CACHE[kind] = docs
    return docs


def _doc_rows(docs: list[dict], max_length: int) -> list[dict]:
    out = []
    for d in docs:
        out.append({"id": d["id"], "question": d["question"], "context": truncate_llama2(d["context"], d["_n"], d["_ends"], max_length), "demos": d["demos"], "answers": d["answers"],
                    "extra": _dump({k: v for k, v in d["extra"].items() if v is not None})})
    return out


def build_infbench(spec: Spec, length: int) -> list[dict]:
    return _doc_rows(_infbench_docs(spec.kind), DOC_TOKENS[spec.name][length])


def _lexsum_docs() -> list[dict]:
    if "multi_lexsum" in _DOC_CACHE:
        return _DOC_CACHE["multi_lexsum"]
    val = _read_jsonl(src_root() / "data/multi_lexsum/multi_lexsum_val.jsonl")             # = validation of allenai/multi_lexsum v20230518 (312 cases with a short summary, same order)
    train = [r for r in _iter_jsonl(_hub_file(SRC_LEXSUM, "releases/v20230518/train.json")) if r.get("summary/short") is not None]
    for x in val:
        x["context"] = "\n\n".join(x["sources"])
        picked = [train[int(i)] for i in hf_permutation(len(train), _demo_seed(x["id"]))[:SHOTS]]
        x["demos"] = ("Example summaries:\n\n" + "\n\n".join("Summary: {}".format(ex["summary/short"]) for ex in picked)
                      + "\n\nNow, write a summary of the following legal documents.\n")
    enc = llama2_encode_many([x["context"] for x in val])
    keep = [i for i, (n, _) in enumerate(enc) if n >= MIN_DOC_TOKENS["multi_lexsum"]]
    sel = keep if len(keep) <= N_QUESTIONS else [keep[int(i)] for i in hf_permutation(len(keep), SEED)[:N_QUESTIONS]]
    docs = []
    for i in sel:
        x = val[i]
        docs.append({"id": _row_id("multi_lexsum", x["id"]), "question": "", "context": x["context"], "demos": x["demos"], "answers": [x["summary/short"]],
                     "extra": {"keypoints": x["summary/short_keypoints"], "expert_summary": x["summary/long"], "src_id": x["id"], "doc_llama2_tokens": enc[i][0]},
                     "_n": enc[i][0], "_ends": enc[i][1]})
    _DOC_CACHE["multi_lexsum"] = docs
    return docs


def build_multi_lexsum(spec: Spec, length: int) -> list[dict]:
    return _doc_rows(_lexsum_docs(), DOC_TOKENS[spec.name][length])


def _nqa_docs() -> list[dict]:
    """HELMET load_narrativeqa (HEAD): the test split shuffled with seed 42, the questions whose story has more than 131,072 Llama-2 tokens, the first N_QUESTIONS of them;
    two demo questions per item from the train split (`shuffle(seed=demo_seed(document id + question))`, the first two)."""
    if "narrativeqa" in _DOC_CACHE:
        return _DOC_CACHE["narrativeqa"]
    import glob

    import pyarrow.parquet as pq

    test_files = sorted(glob.glob(str(_hub_file(SRC_NQA, "data/test-*.parquet"))))
    qa = []
    for f in test_files:
        qa += pq.ParquetFile(f).read(columns=["document.id", "question.text", "answers"]).to_pylist()
    texts: dict[str, str] = {}
    for f in test_files:                                     # one story per document id (the stories repeat in every question row)
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=32, columns=["document.id", "document.text"]):
            for r in batch.to_pylist():
                texts.setdefault(r["document"]["id"], r["document"]["text"])
    ids = list(texts)
    tok_len = dict(zip(ids, llama2_encode_many([texts[d] for d in ids])))
    train = []
    for f in sorted(glob.glob(str(_hub_file(SRC_NQA, "data/train-*.parquet")))):
        train += pq.ParquetFile(f).read(columns=["question.text", "answers"]).to_pylist()
    docs = []
    for i in hf_permutation(len(qa), SEED):
        r = qa[int(i)]
        did = r["document"]["id"]
        n, ends = tok_len[did]
        if n <= MIN_DOC_TOKENS["narrativeqa"]:
            continue
        picked = [train[int(j)] for j in hf_permutation(len(train), _demo_seed(did + r["question"]["text"]))[:SHOTS]]
        demos = ("For example:\n\n" + "\n\n".join(f"Question: {ex['question']['text']}\nAnswer: {ex['answers'][0]['text']}" for ex in picked)
                 + "\n\nNow, use the following story to answer the question:\n\n")
        docs.append({"id": _row_id("narrativeqa", did, len(docs)), "question": r["question"]["text"], "context": texts[did], "demos": demos,
                     "answers": [a["text"] for a in r["answers"]], "extra": {"src_id": did, "doc_llama2_tokens": n}, "_n": n, "_ends": ends})
        if len(docs) == N_QUESTIONS:
            break
    _DOC_CACHE["narrativeqa"] = docs
    return docs


def build_narrativeqa(spec: Spec, length: int) -> list[dict]:
    return _doc_rows(_nqa_docs(), DOC_TOKENS[spec.name][length])


BUILDERS: dict[str, Callable[[Spec, int], list[dict]]] = {"json_kv": build_json_kv, "ruler": build_ruler, "qa": build_qa, "rerank": build_rerank, "icl": build_icl,
                                                          "infbench_qa": build_infbench, "infbench_choice": build_infbench, "infbench_sum": build_infbench,
                                                          "multi_lexsum": build_multi_lexsum, "narrativeqa": build_narrativeqa}


# ---- Llama-2 token counts, manifest
@functools.lru_cache(maxsize=1)
def _llama2_tokenizer():
    from tokenizers import Tokenizer

    snap = _hf_home() / "hub" / "models--NousResearch--Llama-2-7b-hf" / "snapshots" / LLAMA2_TOKENIZER_REVISION
    return Tokenizer.from_file(str(snap / "tokenizer.json"))


def count_tokens_llama2(texts: list[str]) -> list[int]:
    tok = _llama2_tokenizer()
    return [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=True)]      # HELMET: tokenizer([prompt]) adds <s>


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _stats(vals: list[int]) -> dict:
    return {"min": min(vals), "max": max(vals), "mean": round(sum(vals) / len(vals), 1)} if vals else {"min": None, "max": None, "mean": None}


def build_subset(spec: Spec, length: int, builders: dict | None = None) -> Path:
    """Build and save one (length, subset); returns its folder. Atomic: a temp folder, then rename."""
    from datasets import Dataset, DatasetDict

    rows = (builders or BUILDERS)[spec.kind](spec, length)
    if not rows:
        raise ValueError(f"{spec.name} @ {length}: no rows")
    n_tok = count_tokens_llama2([build_prompt(spec, r, chat=False) for r in rows])
    for r, n in zip(rows, n_tok):
        r["n_llama2"] = n
    dest = base_dir(length, spec.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".tmp-{spec.name}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    DatasetDict({"test": Dataset.from_list(rows)}).save_to_disk(str(tmp))
    if dest.exists():
        shutil.rmtree(dest)
    os.rename(tmp, dest)
    _update_manifest(spec, length, rows, n_tok)
    return dest


def _update_manifest(spec: Spec, length: int, rows: list[dict], n_tok: list[int]) -> None:
    path = helmet_root() / "manifest.json"
    man = json.loads(path.read_text()) if path.is_file() else {}
    man.update({
        "schema": 1, "data_tag": DATA_TAG, "updated": _now(), "seed": SEED, "n_questions": N_QUESTIONS, "n_icl": N_ICL, "shots": SHOTS,
        "sources": {
            "helmet_data": {"repo": HELMET_REPO, "revision": HELMET_REVISION, "file": HELMET_TARBALL, "sha256": HELMET_TARBALL_SHA256},
            "helmet_code": {"repo": "github.com/princeton-nlp/HELMET", "commit": HELMET_CODE_COMMIT, "license": HELMET_CODE_LICENSE},
            "llama2_tokenizer": {"repo": LLAMA2_TOKENIZER_REPO, "revision": LLAMA2_TOKENIZER_REVISION},
            "icl": {k: {"repo": v[0], "revision": v[1], "files": v[2]} for k, v in ICL_SOURCES.items()},
            "nlu": {"url": SRC_NLU_URL, "sha256": SRC_NLU_SHA256, "license": "CC BY 4.0 (repo LICENSE)"},
            "long_documents": {k: {"repo": v[0], "revision": v[1], "license": v[2]} for k, v in (("infinitebench", SRC_INFBENCH), ("multi_lexsum", SRC_LEXSUM), ("narrativeqa", SRC_NQA))},
        },
    })
    data = man.setdefault("data", {})
    data.setdefault(str(length), {})[spec.name] = {
        "rows": len(rows), "questions": len({r["question"] for r in rows}), "llama2_tokens": _stats(n_tok), "created": _now(), "category": spec.category,
        "gen": spec.gen, "stop_newline": spec.stop_nl}
    tmp = path.with_suffix(f".json.tmp{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def prepared(length: int, name: str) -> bool:
    return (base_dir(length, name) / "dataset_dict.json").exists()


def missing_pairs(lengths=LENGTHS, names=None) -> list[tuple[int, str]]:
    return [(L, n) for L in lengths for n in (names or AVAILABLE) if not prepared(L, n)]


def prepare(lengths=LENGTHS, names=None, force: bool = False) -> list[tuple[int, str]]:
    """Login-node entry (`titan-eval prepare helmet`): extract what is needed and build every missing (length, subset). Returns the pairs it built."""
    names = list(names or AVAILABLE)
    lengths = tuple(lengths)
    todo = [(L, n) for L in lengths for n in names if force or not prepared(L, n)]
    if not todo:
        return []
    extract(member_files(sorted({L for L, _ in todo}), sorted({n for _, n in todo})))
    for L, n in todo:
        logger.info(f"HELMET: building {n} @ {L}")
        build_subset(SPECS[n], L)
    return todo



# ---- per-tokenizer copies: HELMET's run-time cut of the context to `L - generation length` in the model's own tokens (model_utils.tokenize)
def _load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def encode_lengths(tok, texts: list[str]) -> list[int]:
    """Token counts without special tokens (our models get no BOS from vLLM; HELMET's HF tokenizers add one: at most one token of difference)."""
    bt = getattr(tok, "backend_tokenizer", None)
    if bt is not None:
        return [len(e.ids) for e in bt.encode_batch(texts, add_special_tokens=False)]
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def cut_context(tok, spec: Spec, row: dict, limit: int, n_tokens: int) -> tuple[dict, int, int]:
    """HELMET model_utils.tokenize: when the prompt is longer than `limit`, the END of `context` is dropped by the excess number of context tokens
    (new_context = context[:offset_mapping[-excess][0]]); the prompt is rendered again. A re-tokenisation can leave a few tokens over: the cut is repeated.
    Returns (row, tokens after the cut, tokens removed from the context in total)."""
    row = dict(row)
    removed = 0
    for _ in range(5):
        excess = n_tokens - limit
        if excess <= 0:
            break
        offs = tok(row["context"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        row["context"] = "" if len(offs) <= excess else row["context"][: offs[-excess][0]]
        removed += excess
        n_tokens = encode_lengths(tok, [build_prompt(spec, row, chat=False)])[0]
    if n_tokens > limit:
        raise ValueError(f"{spec.name}: row {row.get('id')} still has {n_tokens} tokens after the cut (limit {limit})")
    return row, n_tokens, removed


def load_base(length: int, name: str) -> list[dict]:
    from datasets import load_from_disk

    return load_from_disk(str(base_dir(length, name)))["test"].to_list()


def tok_manifest(key: str) -> dict:
    p = tok_root(key) / "manifest.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def tok_prepared(key: str, length: int, name: str) -> bool:
    return (tok_dir(key, length, name) / "dataset_dict.json").exists()


def tok_status(tokenizer_path: str, lengths=LENGTHS, names=None) -> dict:
    """Which (length, subset) copies of this tokenizer are missing (reads files only)."""
    key = tokenizer_key(tokenizer_path)
    return {"tokenizer": tokenizer_path, "key": key, "dir": str(tok_root(key)),
            "missing": [(L, n) for L in lengths for n in (names or AVAILABLE) if not tok_prepared(key, L, n)],
            "base_missing": [(L, n) for L in lengths for n in (names or AVAILABLE) if not prepared(L, n)]}


def prepare_tokenizer(tokenizer_path: str, lengths=LENGTHS, names=None, force: bool = False) -> list[tuple[int, str]]:
    """Login node (`titan-eval prepare helmet --model X`): the per-tokenizer copies of the base data. Returns the (length, subset) pairs it wrote."""
    from datasets import Dataset, DatasetDict
    from lighteval.tasks.tasks.ruler import _get_model_arch, _tokenizer_file_digests

    key = tokenizer_key(tokenizer_path)
    names = list(names or AVAILABLE)
    todo = [(L, n) for L in lengths for n in names if force or not tok_prepared(key, L, n)]
    if not todo:
        return []
    absent = [(L, n) for L, n in todo if not prepared(L, n)]
    if absent:
        raise FileNotFoundError(f"the base data are not prepared for {absent[:3]}...: run `titan-eval prepare helmet` first")
    tok = _load_tokenizer(tokenizer_path)
    for L, n in todo:
        spec = SPECS[n]
        rows = load_base(L, n)
        counts = encode_lengths(tok, [build_prompt(spec, r, chat=False) for r in rows])
        limit = L - spec.gen
        out, cut_rows, cut_total = [], 0, 0
        for r, c in zip(rows, counts):
            removed = 0
            if c > limit:
                r, c, removed = cut_context(tok, spec, r, limit, c)
                cut_rows += 1
                cut_total += removed
            r["n_tokens"], r["cut_tokens"] = c, removed
            out.append(r)
        dest = tok_dir(key, L, n)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".tmp-{n}-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        DatasetDict({"test": Dataset.from_list(out)}).save_to_disk(str(tmp))
        if dest.exists():
            shutil.rmtree(dest)
        os.rename(tmp, dest)
        man = tok_manifest(key)
        man.update({"schema": 1, "key": key, "tokenizer": {"path": str(tokenizer_path), "name": os.path.basename(os.path.normpath(str(tokenizer_path))),
                                                           "architecture": _get_model_arch(tokenizer_path), "files_sha256": _tokenizer_file_digests(tokenizer_path)},
                    "data_tag": DATA_TAG, "updated": _now()})
        man.setdefault("created", man["updated"])
        man.setdefault("data", {}).setdefault(str(L), {})[n] = {
            "rows": len(out), "tokens": _stats([r["n_tokens"] for r in out]), "limit": limit, "gen": spec.gen, "cut_rows": cut_rows,
            "cut_tokens_mean": round(cut_total / cut_rows, 1) if cut_rows else 0.0, "llama2_tokens": _stats([r["n_llama2"] for r in out]), "created": _now()}
        tmpm = tok_root(key) / f"manifest.json.tmp{os.getpid()}"
        tmpm.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        os.replace(tmpm, tok_root(key) / "manifest.json")
        logger.info(f"HELMET {n} @ {L}: {len(out)} rows, mean {man['data'][str(L)][n]['tokens']['mean']} tokens, {cut_rows} cut")
    return todo

AVAILABLE: tuple[str, ...] = RULE_SCORED      # subsets with a builder, a scorer and a registry entry (JUDGE_SCORED are built as data but have no task until the judge stage is wired)

TASKS_TABLE = get_helmet_tasks(os.environ.get("TOKENIZER_PATH"))
