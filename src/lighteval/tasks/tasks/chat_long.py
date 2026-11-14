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

"""Long-context chat evals (titan_eval, post-trained models; plan chat_eval section 3.7). Module of chat_long.py in the fork.

longbench_v2_chat: LongBench v2 (zai-org/LongBench-v2, 503 four-choice questions, contexts 8k to 2M words, Apache-2.0).
  Prompt = the official zero-shot template (THUDM/LongBench prompts/0shot.txt, "The correct answer is (X)") with the
  context, the question and the four choices. The official truncation rule (THUDM/LongBench pred.py): the tokenised prompt
  is cut in the MIDDLE, keeping the first half and the last half of the budget, so the question at the end survives.
  Here the budget is the model's context minus the generation budget minus a template margin:
      prompt budget = TITAN_EVAL_MAX_MODEL_LEN - TITAN_EVAL_MAX_NEW_TOKENS - CHAT_MARGIN
  The run exports the three TITAN_EVAL_* variables (titan_eval/evals/adapters/lighteval.py, chat_render.py); the tokenizer is the
  model's own (TITAN_EVAL_TOKENIZER). Without them the prompt is not cut (a run then fails in vLLM, loudly).
  Score: accuracy of the anchored "Answer: X" / "The correct answer is (X)" in the text after the last </think> (the chat metric
  of chat_metrics.py; the official extract_answer takes the same regex). Overall `acc` plus the official breakdowns: difficulty
  easy / hard and length short / medium / long. A breakdown metric is NaN for the items outside its bucket and is averaged over
  its own members (np.nanmean), so one run gives the overall and the five breakdowns.
  The official CoT variant (0shot_cot + a second answer call) is not built: a thinking model writes its own reasoning block.
"""

import functools
import os
from string import ascii_uppercase

import numpy as np

from lighteval.metrics.chat_metrics import ChatSamples, _choice_scores, _letters_extractor
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


BUDGET = 32768                # default max new tokens (the run's gen.max_new_tokens replaces it; chat_defaults)
CHAT_MARGIN = 256             # tokens kept free for the chat template, the answer instruction and the re-tokenisation after the cut
N_SAMPLES = 1                 # one sample per question (the official protocol is single-sample; the run's num_samples may raise it)

LB2_REPO = "zai-org/LongBench-v2"
LB2_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"   # HF commit of the dataset (the repo moved from THUDM/ to zai-org/); data.json, 503 rows

# THUDM/LongBench prompts/0shot.txt, byte for byte (the placeholders are replaced in the same order as pred.py)
LB2_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n$DOC$\n</text>\n\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n\n"
    "Format your response as follows: \"The correct answer is (insert answer here)\"."
)

ENV_MAX_MODEL_LEN = "TITAN_EVAL_MAX_MODEL_LEN"
ENV_MAX_NEW_TOKENS = "TITAN_EVAL_MAX_NEW_TOKENS"
ENV_TOKENIZER = "TITAN_EVAL_TOKENIZER"


def prompt_budget() -> int | None:
    """Tokens the prompt may take (template included): context - generation - margin. None when the run did not export the context."""
    ctx, new = os.environ.get(ENV_MAX_MODEL_LEN), os.environ.get(ENV_MAX_NEW_TOKENS)
    if not ctx or not new:
        return None
    return int(ctx) - int(new) - CHAT_MARGIN


@functools.lru_cache(maxsize=4)
def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def truncate_middle(text: str, budget: int, tok) -> str:
    """THUDM/LongBench pred.py: ids[:budget//2] + ids[-(budget - budget//2):] (the question at the end is kept), decoded without special tokens."""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text
    keep = ids[: budget // 2] + ids[-(budget - budget // 2):]
    return tok.decode(keep, skip_special_tokens=True)


def longbench_v2_prompt(line, task_name: str = None):
    prompt = (
        LB2_TEMPLATE.replace("$DOC$", line["context"].strip())
        .replace("$Q$", line["question"].strip())
        .replace("$C_A$", line["choice_A"].strip())
        .replace("$C_B$", line["choice_B"].strip())
        .replace("$C_C$", line["choice_C"].strip())
        .replace("$C_D$", line["choice_D"].strip())
    )
    budget = prompt_budget()
    tok_path = os.environ.get(ENV_TOKENIZER)
    if budget is not None and tok_path:
        if budget <= 0:
            raise ValueError(f"longbench_v2_chat: no room for the prompt (context - generation - margin = {budget})")
        prompt = truncate_middle(prompt, budget, _tokenizer(tok_path))
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=list(ascii_uppercase[:4]),
        gold_index=ascii_uppercase.index(line["answer"]),
        specific={
            "_id": line["_id"],
            "domain": line["domain"],
            "sub_domain": line["sub_domain"],
            "difficulty": line["difficulty"],
            "length": line["length"],
        },
    )


# ------------------------------------------------------------------------------------------------------------- metric
BUCKETS = {
    "acc_easy": ("difficulty", "easy"),
    "acc_hard": ("difficulty", "hard"),
    "acc_short": ("length", "short"),
    "acc_medium": ("length", "medium"),
    "acc_long": ("length", "long"),
}
NAMES = ("acc", "acc_unanchored", *BUCKETS)

_base_score = _choice_scores(_letters_extractor())   # acc (anchored letter) and acc_unanchored (diagnostic), chat_metrics


def _score(doc: Doc, pred: str) -> dict[str, float]:
    out = dict(_base_score(doc, pred))
    spec = doc.specific or {}
    for name, (field, value) in BUCKETS.items():
        out[name] = out["acc"] if spec.get(field) == value else float("nan")
    return out


def _members_avg(values) -> float:
    """Mean over the members of a bucket (NaN = not a member). The name has no "mean" on purpose: lighteval gives "mean" aggregations the plain
    stderr, which NaN would poison, and bootstraps every other aggregation."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def longbench_v2_metric() -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
        metric_name=list(NAMES),
        sample_level_fn=ChatSamples(N_SAMPLES, _score, NAMES, ks=()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={name: (np.mean if name in ("acc", "acc_unanchored") else _members_avg) for name in NAMES},
        higher_is_better={name: True for name in NAMES},
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="longbench_v2_chat",
        prompt_function=longbench_v2_prompt,
        hf_repo=LB2_REPO,
        hf_subset="default",
        hf_revision=LB2_REVISION,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[longbench_v2_metric()],
        version=0,
    ),
]


# ----------------------------------------------------------------------------------------------------------- LongBench Pro
# LongBench Pro (caskcsg/LongBench-Pro @ 4996884d, Apache-2.0, arXiv 2601.02872): 1,500 bilingual (EN / ZH) questions, contexts 8k to 256k
# tokens, 11 primary / 25 secondary tasks. Ground-truth scoring (no LLM judge), code: github caskcsg/longcontext LongBench-Pro (modules/utils.py,
# inference.py, model_manager.py). Prompt = context + "\n\n\n\n" + the question (question_thinking when thinking is on, else
# question_nonthinking); the official middle truncation of the whole prompt (model_manager._truncate_prompt: ids[:L/2] + ids[-L/2:]) with the
# prompt budget of this module. Metric per secondary task (official `task_metric_config`): NDCG (T1.x, reimplemented: pytrec_eval is not
# installed), pairwise accuracy (T2.x, T6.3), accuracy (T3.x, T11.x), F1 (T5.x, T6.2, T7.x, T9.x), SubEM (T6.1, T8.x, T10.x), and Summary (T4.x) =
# 0.5 x max cosine similarity to the answer points (Qwen3-Embedding-8B, sentence_transformers) + 0.5 x max ROUGE-L (the `rouge` package, jieba
# for Chinese). Summary needs the embedding model: TITAN_EVAL_EMBEDDING_MODEL (a folder; not on disk, login-node download) and the `rouge`
# package (not installed): a run with a T4 item fails loudly until both are there.
LBP_REPO = "caskcsg/LongBench-Pro"
LBP_REVISION = "4996884deae51f5e5d23c88da9d857fc54e5fa15"   # HF commit of the dataset (longbench_pro.json, 1,500 rows, test split)
ENV_THINKING = "TITAN_EVAL_THINKING"                         # "1" thinking on (question_thinking), "0" off (question_nonthinking); unset = on (Qwen3 default)
ENV_EMBEDDING = "TITAN_EVAL_EMBEDDING_MODEL"

LBP_DIMS = {
    "difficulty": ["Easy", "Moderate", "Hard", "Extreme"],
    "token_length": ["8k", "16k", "32k", "64k", "128k", "256k"],
    "language": ["English", "Chinese"],
    "contextual_requirement": ["Full", "Partial"],
    "primary_task": [f"T{i}" for i in range(1, 12)],
}
LBP_BUCKETS = {f"acc_{v.lower()}": (dim, v) for dim, vals in LBP_DIMS.items() for v in vals}
LBP_NAMES = ("acc", *LBP_BUCKETS)

_LBP_METRIC = {}     # secondary task -> metric name (the official task_metric_config, LongBench-Pro modules/evaluation.py)
for _k in ("T1.1", "T1.2"):
    _LBP_METRIC[_k] = "NDCG"
for _k in ("T2.1", "T2.2", "T6.3"):
    _LBP_METRIC[_k] = "Pairwise_Accuracy"
for _k in ("T3.1", "T3.2", "T11.1", "T11.2"):
    _LBP_METRIC[_k] = "Accuracy"
for _k in ("T5.1", "T5.2", "T6.2", "T7.1", "T7.2", "T7.3", "T9.1", "T9.2"):
    _LBP_METRIC[_k] = "F1_Score"
for _k in ("T6.1", "T8.1", "T8.2", "T8.3", "T10.1", "T10.2"):
    _LBP_METRIC[_k] = "SubEM"
for _k in ("T4.1", "T4.2"):
    _LBP_METRIC[_k] = "Summary"


def lbp_prompt(line, task_name: str = None):
    thinking = os.environ.get(ENV_THINKING, "1") != "0"
    question = line["question_thinking"] if thinking else line["question_nonthinking"]
    prompt = f"{line['context']}\n\n\n\n{question}".strip()
    budget = prompt_budget()
    tok_path = os.environ.get(ENV_TOKENIZER)
    if budget is not None and tok_path:
        if budget <= 0:
            raise ValueError(f"longbench_pro_chat: no room for the prompt (context - generation - margin = {budget})")
        prompt = truncate_middle(prompt, budget, _tokenizer(tok_path))
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[""],                  # the gold is `answer` in specific (ChatSamples needs one gold; the scorer reads specific)
        gold_index=0,
        specific={
            "id": line["id"], "answer": list(line["answer"]), "secondary_task": line["secondary_task"], "language": line["language"],
            "difficulty": line["difficulty"], "token_length": line["token_length"], "contextual_requirement": line["contextual_requirement"],
            "primary_task": line["primary_task"].split(".")[0].strip(),
        },
    )


# ---- the official scoring functions (LongBench-Pro modules/utils.py), reimplemented where the package is missing
def _get_answer_area(text: str) -> str:
    if "[Answer]" in text or "[答案]" in text:
        if "[Answer]" in text:
            text = text[text.rfind("[Answer]") + 8:]
        else:
            text = text[text.rfind("[答案]") + 4:]
    return text.strip()


def _fix_space(text: str) -> str:
    return " ".join(text.split())


def _norm_answers(answers):
    return [_fix_space(a.lower().strip()) for a in answers]


def _norm_prediction(prediction: str):
    # official order: the answer area first (the "[Answer]" tag is case-sensitive), then lower case
    return [_fix_space(p.strip()) for p in _get_answer_area(prediction).lower().split("\n")]


def lbp_accuracy(answers, prediction):
    answers, predictions = _norm_answers(answers), _norm_prediction(prediction)
    if not answers or not predictions:
        return 0.0
    return 1.0 if answers[0] == predictions[0] else 0.0


def lbp_f1(answers, prediction):
    answers, predictions = set(_norm_answers(answers)), set(_norm_prediction(prediction))
    common = answers & predictions
    if not common or not predictions or not answers:
        return 0.0
    precision, recall = len(common) / len(predictions), len(common) / len(answers)
    return 2 * precision * recall / (precision + recall)


def lbp_subem(answers, prediction):
    answers, predictions = _norm_answers(answers), _norm_prediction(prediction)
    if not answers or not predictions:
        return 0.0
    return sum(1.0 for a in answers if a in predictions) / len(answers)


def lbp_pairwise(answers, prediction):
    from itertools import combinations

    answers, predictions = _norm_answers(answers), _norm_prediction(prediction)
    if len(answers) <= 1 or len(predictions) <= 1:
        return 0.0
    n_total = len(predictions) * (len(predictions) - 1) // 2
    index = {p: i for i, p in enumerate(predictions)}
    n_correct = sum(1 for a, b in combinations(answers, 2) if a in index and b in index and index[a] < index[b])
    return n_correct / n_total


def lbp_ndcg(answers, prediction):
    """pytrec_eval's ndcg_cut.<len(answers)> as the official code calls it: qrels rel = len - i (gain = rel, discount 1 / log2(rank + 1)),
    the run ranks the prediction lines by their position, lines that are not answers have gain 0, IDCG over all qrels."""
    import math

    answers, predictions = _norm_answers(answers), _norm_prediction(prediction)
    if not answers or not predictions:
        return 0.0
    k = len(answers)
    qrels = {a: len(answers) - i for i, a in enumerate(answers)}
    ranking = [p for p in predictions]                               # the run: earlier line = higher score
    gains = [qrels.get(p, 0) for p in ranking][:k]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


_EMBEDDER = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        path = os.environ.get(ENV_EMBEDDING)
        if not path:
            raise RuntimeError(f"longbench_pro_chat Summary (T4) needs the embedding model: set {ENV_EMBEDDING} to the Qwen3-Embedding-8B folder "
                               "(login-node download; not on disk)")
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(path, tokenizer_kwargs={"padding_side": "left"})
    return _EMBEDDER


def lbp_summary(answers, prediction, is_zh: bool, alpha: float = 0.5, beta: float = 0.5):
    answers = _norm_answers(answers)
    prediction = _fix_space(prediction.lower().strip())
    if not answers or not prediction:
        return 0.0
    model = _embedder()
    a_emb = model.encode(answers)
    p_emb = model.encode([prediction])
    semantic = float(model.similarity(a_emb, p_emb).max().cpu().numpy())
    if is_zh:
        import jieba

        answers_t = [" ".join(jieba.cut(a, cut_all=False)) for a in answers]
        pred_t = " ".join(jieba.cut(prediction, cut_all=False))
    else:
        answers_t, pred_t = answers, prediction
    try:
        from rouge import Rouge
    except ImportError as e:
        raise RuntimeError("longbench_pro_chat Summary (T4) needs the `rouge` package (not installed; owner installs)") from e
    try:
        scores = Rouge().get_scores([pred_t] * len(answers_t), answers_t, avg=False)
        rouge_l = max(s["rouge-l"]["f"] for s in scores)
    except Exception:
        rouge_l = 0.0
    return alpha * semantic + beta * rouge_l


def lbp_metric_name(secondary: str) -> str:
    """The official metric of a secondary task. The dataset holds the full name ("T1.1 Global Cohesive Retrieval"): the code ("T1.1") is its first word."""
    return _LBP_METRIC[secondary.split(" ")[0]]


def lbp_metric_value(secondary: str, answers, prediction: str, is_zh: bool) -> float:
    if not prediction:
        return 0.0
    name = lbp_metric_name(secondary)
    if name == "NDCG":
        return lbp_ndcg(answers, prediction)
    if name == "Pairwise_Accuracy":
        return lbp_pairwise(answers, prediction)
    if name == "Accuracy":
        return lbp_accuracy(answers, prediction)
    if name == "F1_Score":
        return lbp_f1(answers, prediction)
    if name == "SubEM":
        return lbp_subem(answers, prediction)
    return lbp_summary(answers, prediction, is_zh)


def _lbp_score(doc: Doc, pred: str) -> dict[str, float]:
    spec = doc.specific
    value = lbp_metric_value(spec["secondary_task"], spec["answer"], pred.strip(), spec["language"] == "Chinese") if pred and pred.strip() else 0.0
    out = {"acc": value}
    for name, (dim, v) in LBP_BUCKETS.items():
        out[name] = value if spec.get(dim) == v else float("nan")
    return out


def longbench_pro_metric() -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
        metric_name=list(LBP_NAMES),
        sample_level_fn=ChatSamples(N_SAMPLES, _lbp_score, LBP_NAMES, ks=()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={name: (np.mean if name == "acc" else _members_avg) for name in LBP_NAMES},
        higher_is_better={name: True for name in LBP_NAMES},
    )


TASKS_TABLE += [
    LightevalTaskConfig(
        name="longbench_pro_chat",
        prompt_function=lbp_prompt,
        hf_repo=LBP_REPO,
        hf_subset="default",
        hf_revision=LBP_REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[longbench_pro_metric()],
        version=0,
    ),
]
