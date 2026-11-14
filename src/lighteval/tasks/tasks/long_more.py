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

"""More long-context evals beyond RULER (titan_eval, owner 11-14; plan chat_eval 3.7). Base AND chat variants, one task per context length where the benchmark has lengths.

longbench_v2_base_<L>   LongBench v2 (zai-org/LongBench-v2, 503 four-choice questions) for BASE checkpoints, L = 8192 .. 131072: the official zero-shot template without a chat template,
                        the official middle truncation of the whole prompt to L (the question and the choices stay at the end), then the answer prefix "The correct answer is (" and a greedy
                        generation of a few tokens: the first letter A-D. Metrics: acc, the official breakdowns (easy / hard, short / medium / long words), token buckets of the untruncated prompt,
                        acc_untruncated / acc_truncated and the share of cut prompts. (The chat twin is `longbench_v2_chat`, module chat_long.py.)
gsm_infinite_<hard|medium>_<L>[_chat]   GSM-Infinite (InfiniAILab, arXiv 2502.05252): noise-filled reasoning problems, L = 8192 .. 131072 (the dataset's 8k .. 128k), op counts 2 .. 20, accuracy per
                        op count. Data: the prepared sample of titan_eval/evals/long_data.py (official `limit` rule, 20 per op). Chat = the dataset's messages as given; base = the same text as plain
                        text (instruction, then "Problem: ... Question: ... Solution:"), zero-shot (the paper has no base protocol). Answer = the official parser (eval_realistic.py criteriaoutput).
mrcr_<L>[_chat]         OpenAI MRCR (openai/mrcr, MIT): 2 / 4 / 8 needles, the bin of L = 8192 .. 262144 tokens (prompt + answer), 100 rows per needle count. Chat = the conversation as given; base = the
                        same conversation as a "User: / Assistant:" transcript ending in "Assistant:". Official grader verbatim (prefix check, then SequenceMatcher ratio). Items whose prompt does not fit
                        the run's context are skipped as in the official script (counted: `skipped`).
frames_chat             FRAMES (google/frames-benchmark, 824 multi-hop questions), ORACLE setting: all linked Wikipedia articles (parasail-ai/frames-benchmark-wikipedia, the pinned Oct-2024 dump) in the
                        prompt, middle-truncated to the context. Judge-graded: the lighteval metric is `answered`, the accuracy comes from the judge stage (titan_eval/evals/judge.py, grader `frames`
                        = the paper's auto-rating prompt, Figure 6).

The budget of every prompt = the run's context (TITAN_EVAL_MAX_MODEL_LEN, the task's vllm.max_model_len) minus the generation cap minus a margin; the tokenizer is the model's own
(TITAN_EVAL_TOKENIZER). Without these variables nothing is cut or skipped (unit tests, `lighteval tasks inspect`).
"""

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from string import ascii_uppercase

import numpy as np

from lighteval.metrics.chat_metrics import ChatSamples
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.chat_long import (
    ENV_MAX_MODEL_LEN,
    ENV_MAX_NEW_TOKENS,
    ENV_TOKENIZER,
    LB2_REPO,
    LB2_REVISION,
    LB2_TEMPLATE,
    _members_avg,
    _tokenizer,
)


N_SAMPLES = 1
NAN = float("nan")
CHAT_PER_MESSAGE = 6       # tokens a chat template adds around one message (ChatML 5, Llama-3 header 5); counted for conversations with many turns
SKIP_USER = "Reply with the word OK."      # the prompt of an item that does not fit the context: skipped (NaN metrics), kept tiny so the run does not spend anything on it


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    return int(v) if v else None


def run_context(default: int | None = None) -> int | None:
    """The run's context window (vllm.max_model_len exported by the adapter); `default` when the run did not export it."""
    return _env_int(ENV_MAX_MODEL_LEN) or default


def run_new_tokens(default: int) -> int:
    """The generation cap of the run (gen.max_new_tokens), else the task's own."""
    return _env_int(ENV_MAX_NEW_TOKENS) or default


def xdg_dir() -> str:
    """<cache>/xdg: where titan_eval/evals/long_data.py puts the prepared files (the IMO-AnswerBench layout). $TITAN_EVAL_CACHE, else the parent of $HF_HOME."""
    root = os.environ.get("TITAN_EVAL_CACHE") or (str(Path(os.environ["HF_HOME"]).parent) if os.environ.get("HF_HOME") else "")
    return str(Path(root) / "xdg") if root else ""


def prepared(name: str) -> str:
    """The folder of one prepared dataset (hf_repo of the lighteval task)."""
    return str(Path(xdg_dir() or ".") / name)


def _nan_if(member: bool, value: float) -> float:
    return value if member else NAN


# ======================================================================================================== LongBench v2, base
LB2B_PREFIX = "The correct answer is ("          # appended after the official template: the model continues with the letter
LB2B_LENGTHS = (8192, 16384, 32768, 65536, 131072)
LB2B_NEW = 8                                    # generated tokens (the letter and the closing bracket)
LB2B_MARGIN = 32                                # tokens kept free for the re-tokenisation after the cut

LB2B_TOKEN_BUCKETS = {                          # untruncated prompt tokens (the run's tokenizer): (lower, upper]
    "acc_tok_0_16k": (0, 16384),
    "acc_tok_16k_32k": (16384, 32768),
    "acc_tok_32k_64k": (32768, 65536),
    "acc_tok_64k_128k": (65536, 131072),
    "acc_tok_128k_up": (131072, float("inf")),
}
LB2B_OFFICIAL = {"acc_easy": ("difficulty", "easy"), "acc_hard": ("difficulty", "hard"), "acc_short": ("length", "short"), "acc_medium": ("length", "medium"),
                 "acc_long": ("length", "long")}
LB2B_NAMES = ("acc", *LB2B_OFFICIAL, "acc_untruncated", "acc_truncated", *LB2B_TOKEN_BUCKETS, "truncated", "no_letter")


def lb2_base_filled(line) -> str:
    """The official zero-shot prompt (THUDM/LongBench prompts/0shot.txt, same fill order and strips as chat_long.longbench_v2_prompt)."""
    return (
        LB2_TEMPLATE.replace("$DOC$", line["context"].strip())
        .replace("$Q$", line["question"].strip())
        .replace("$C_A$", line["choice_A"].strip())
        .replace("$C_B$", line["choice_B"].strip())
        .replace("$C_C$", line["choice_C"].strip())
        .replace("$C_D$", line["choice_D"].strip())
    )


def cut_middle(text: str, budget: int, tok) -> tuple[str, int, bool]:
    """THUDM/LongBench pred.py rule on the token ids: ids[:budget // 2] + ids[-(budget - budget // 2):], decoded without special tokens. -> (text, tokens before the cut, cut?)"""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text, len(ids), False
    keep = ids[: budget // 2] + ids[-(budget - budget // 2):]
    return tok.decode(keep, skip_special_tokens=True), len(ids), True


def lb2_base_prompt_fn(nominal: int):
    def prompt(line, task_name: str = None):
        text = lb2_base_filled(line)
        tail = "\n" + LB2B_PREFIX
        n_full, cut = -1, False
        tok_path = os.environ.get(ENV_TOKENIZER)
        if tok_path:
            tok = _tokenizer(tok_path)
            n_tail = len(tok.encode(tail, add_special_tokens=False))
            budget = (run_context(nominal) or nominal) - run_new_tokens(LB2B_NEW) - LB2B_MARGIN - n_tail
            if budget <= 0:
                raise ValueError(f"longbench_v2_base_{nominal}: no room for the prompt (context - generation - margin = {budget})")
            text, n_text, cut = cut_middle(text, budget, tok)
            n_full = n_text + n_tail
        return Doc(
            task_name=task_name,
            query=text + tail,
            choices=list(ascii_uppercase[:4]),
            gold_index=ascii_uppercase.index(line["answer"]),
            specific={"_id": line["_id"], "domain": line["domain"], "sub_domain": line["sub_domain"], "difficulty": line["difficulty"], "length": line["length"],
                      "n_tokens": n_full, "truncated": cut},
        )

    return prompt


_LETTER = re.compile(r"\s*\(?\s*([A-D])(?![A-Za-z0-9])")


def lb2_base_letter(generated: str) -> str | None:
    """The letter the model wrote after the forced prefix "The correct answer is (": "B)", "B", " B." -> B. Anything else (a word, a digit, empty) is no answer."""
    m = _LETTER.match(generated or "")
    return m.group(1) if m else None


def _lb2_base_score(doc: Doc, pred: str) -> dict[str, float]:
    spec = doc.specific or {}
    gold = doc.choices[doc.gold_index if isinstance(doc.gold_index, int) else doc.gold_index[0]]
    letter = lb2_base_letter(pred)
    acc = 1.0 if letter == gold else 0.0
    n_tokens, cut = spec.get("n_tokens", -1), spec.get("truncated")
    out = {"acc": acc, "no_letter": 0.0 if letter else 1.0, "truncated": float(bool(cut)) if n_tokens >= 0 else NAN}
    for name, (field, value) in LB2B_OFFICIAL.items():
        out[name] = _nan_if(spec.get(field) == value, acc)
    out["acc_untruncated"] = _nan_if(n_tokens >= 0 and not cut, acc)
    out["acc_truncated"] = _nan_if(n_tokens >= 0 and bool(cut), acc)
    for name, (lo, hi) in LB2B_TOKEN_BUCKETS.items():
        out[name] = _nan_if(n_tokens >= 0 and lo < n_tokens <= hi, acc)
    return out


def longbench_v2_base_metric() -> SampleLevelMetricGrouping:
    plain = ("acc", "no_letter")
    return SampleLevelMetricGrouping(
        metric_name=list(LB2B_NAMES),
        sample_level_fn=ChatSamples(N_SAMPLES, _lb2_base_score, LB2B_NAMES, ks=()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={name: (np.mean if name in plain else _members_avg) for name in LB2B_NAMES},
        higher_is_better={name: name not in ("no_letter", "truncated") for name in LB2B_NAMES},
    )


def _lb2_base_config(length: int) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"longbench_v2_base_{length}",
        prompt_function=lb2_base_prompt_fn(length),
        hf_repo=LB2_REPO,
        hf_subset="default",
        hf_revision=LB2_REVISION,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=LB2B_NEW,
        stop_sequence=["\n"],
        metrics=[longbench_v2_base_metric()],
        version=0,
    )


TASKS_TABLE = [_lb2_base_config(n) for n in LB2B_LENGTHS]


# ======================================================================================================== GSM-Infinite
GSM_SUBSETS = ("hard", "medium")
GSM_LABEL = {8192: "8k", 16384: "16k", 32768: "32k", 65536: "64k", 131072: "128k"}
GSM_OPS = tuple(range(2, 21))
GSM_BASE_NEW = 2048                     # base generation cap (the paper's 4,096 is for chat APIs; the dataset's own solutions are 30 tokens per op)
GSM_CHAT_NEW = 8192                     # chat default cap (the registry yaml sets it; thinking models may need more)
GSM_MARGIN = 64
GSM_BASE_STOP = ["\nProblem:"]         # a base model that has answered goes on with a new "Problem:"

_GSM_KEYWORDS = ["answer: ", "solution: ", "oxed{", "**answer:** ", "**answer: ", "final answer: answer: ", "\nanswer: ", r"\text{answer: } ", "is ", "answer: "]
_GSM_KEYWORDS_END = [".", ".", "}", ".", "**", ".", ".", None, ".", "\n"]


def _is_integer(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def gsm_extract(generated: str) -> int | None:
    """The answer of one reply, the official parser of gsm_infinite/pred/eval_realistic.py (`criteriaoutput`, commit caa53880) ported line by line: lower-case the text, try the keywords in the
    official order (first occurrence; the LAST one for "oxed{" and "is "), cut at the keyword's terminator, accept the first candidate that is an integer. Only change: a digit run that reaches the end
    of the text no longer raises IndexError (the official code would crash on `\\text{answer: } 12`). None = no answer found."""
    text = re.sub(".\x08", "b", generated).lower()
    begin = conclude = -1
    cnt = 0
    while not (begin != -1 and conclude != -1) and cnt < len(_GSM_KEYWORDS):
        kw = _GSM_KEYWORDS[cnt]
        begin = text.rfind(kw) if kw in ("oxed{", "is ") else text.find(kw)
        if begin != -1:
            if _GSM_KEYWORDS_END[cnt] is None:
                conclude = begin + len(kw)
                while conclude < len(text) and text[conclude].isdigit():
                    conclude += 1
            else:
                conclude = text.find(_GSM_KEYWORDS_END[cnt], begin + len(kw))
            if conclude == -1:
                conclude = len(text)
        cnt += 1
        if not _is_integer(text[begin + len(_GSM_KEYWORDS[cnt - 1]):conclude]):
            begin = conclude = -1
            continue
    if begin == -1:
        return None
    try:
        return int(text[begin + len(_GSM_KEYWORDS[cnt - 1]):conclude])
    except ValueError:
        return None


def _fit(n_prompt: int, new: int, margin: int) -> bool:
    """True when prompt + generation cap + margin fit the run's context (always true when the run did not export it)."""
    ctx = run_context()
    return True if ctx is None else n_prompt + new + margin <= ctx


def _count_messages(messages: list[dict], tok) -> int:
    return sum(len(tok.encode(m["content"], add_special_tokens=False)) for m in messages) + CHAT_PER_MESSAGE * len(messages)


def gsm_doc(line, task_name: str, subset: str, chat: bool) -> Doc:
    messages = json.loads(line["messages"])
    skipped, n_prompt = False, -1
    tok_path = os.environ.get(ENV_TOKENIZER)
    if tok_path:
        tok = _tokenizer(tok_path)
        if chat:
            n_prompt = _count_messages(messages, tok)
        else:
            n_prompt = len(tok.encode(_gsm_plain(messages), add_special_tokens=False))
        skipped = not _fit(n_prompt, run_new_tokens(GSM_CHAT_NEW if chat else GSM_BASE_NEW), GSM_MARGIN)
    spec = {"id": line["id"], "op": int(line["op"]), "template": line["template"], "mode": line["mode"], "n_tokens": n_prompt, "skipped": skipped, "subset": subset}
    if skipped:
        spec["chat_messages"] = [{"role": "user", "content": SKIP_USER}] if chat else None
        query = SKIP_USER
    elif chat:
        spec["chat_messages"] = messages
        query = messages[-1]["content"]
    else:
        query = _gsm_plain(messages)
    return Doc(task_name=task_name, query=query, choices=[str(int(line["answer"]))], gold_index=0, specific={k: v for k, v in spec.items() if v is not None})


def _gsm_plain(messages: list[dict]) -> str:
    """Base rendering: the instruction, a blank line, then the dataset's own completion-style turn ("Problem: ... Question: ... Solution:"), without the chat system message and without
    the trailing space of the dataset's text (a base model's first token carries its own space)."""
    users = [m["content"] for m in messages if m["role"] == "user"]
    return "\n\n".join(users).rstrip()


def _gsm_score(doc: Doc, pred: str) -> dict[str, float]:
    spec = doc.specific or {}
    if spec.get("skipped"):
        return {name: NAN for name in GSM_NAMES if name != "skipped"} | {"skipped": 1.0}
    gold = int(doc.get_golds()[0])
    parsed = gsm_extract(pred) if pred and pred.strip() else None
    ok = 1.0 if parsed is not None and parsed == gold else 0.0
    op = spec.get("op")
    out = {"acc": ok, "no_answer": 0.0 if parsed is not None else 1.0, "skipped": 0.0,
           "acc_forward": _nan_if(spec.get("mode") == "normalforward", ok), "acc_reverse": _nan_if(spec.get("mode") == "forwardreverse", ok)}
    for n in GSM_OPS:
        out[f"acc_op{n}"] = _nan_if(op == n, ok)
    return out


GSM_NAMES = ("acc", "acc_forward", "acc_reverse", "no_answer", "skipped", *[f"acc_op{n}" for n in GSM_OPS])


def gsm_metric() -> SampleLevelMetricGrouping:
    return SampleLevelMetricGrouping(
        metric_name=list(GSM_NAMES),
        sample_level_fn=ChatSamples(N_SAMPLES, _gsm_score, GSM_NAMES, ks=()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={name: (np.mean if name == "skipped" else _members_avg) for name in GSM_NAMES},
        higher_is_better={name: name not in ("no_answer", "skipped") for name in GSM_NAMES},
    )


def _gsm_config(subset: str, length: int, chat: bool) -> LightevalTaskConfig:
    def prompt(line, task_name: str = None):
        return gsm_doc(line, task_name, subset, chat)

    name = f"gsm_infinite_{subset}_{length}" + ("_chat" if chat else "")
    return LightevalTaskConfig(
        name=name,
        prompt_function=prompt,
        hf_repo=prepared(f"gsm_infinite_{subset}_{GSM_LABEL[length]}"),
        hf_subset="default",
        hf_revision=None,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=GSM_CHAT_NEW if chat else GSM_BASE_NEW,
        stop_sequence=[] if chat else GSM_BASE_STOP,
        metrics=[gsm_metric()],
        version=0,
    )


TASKS_TABLE += [_gsm_config(s, n, chat) for s in GSM_SUBSETS for n in GSM_LABEL for chat in (False, True)]


# ============================================================================================================== MRCR
MRCR_BINS = (8192, 16384, 32768, 65536, 131072, 262144)
MRCR_NEEDLES = (2, 4, 8)
MRCR_BASE_NEW = 1024                    # the gold answers are up to 800 tokens (Qwen tokenizer; max 3,563 characters)
MRCR_CHAT_NEW = 8192
MRCR_MARGIN = 64
MRCR_BASE_STOP = ["\n\nUser:"]


def mrcr_grade(response, answer, random_string_to_prepend) -> float:
    """The official grader of the dataset card (openai/mrcr README, `grade`), verbatim: the response must start with the random string (no strip); the string is removed from both, then the
    SequenceMatcher ratio of the rest."""
    if not response.startswith(random_string_to_prepend):
        return 0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def mrcr_plain(messages: list[dict]) -> str:
    """Base rendering of the conversation: "User: <text>" / "Assistant: <text>" turns separated by a blank line, ending in "Assistant:" (the dataset card writes its own example this way)."""
    turns = [("User" if m["role"] == "user" else "Assistant") + ": " + m["content"] for m in messages]
    return "\n\n".join(turns) + "\n\nAssistant:"


def mrcr_doc(line, task_name: str, chat: bool) -> Doc:
    messages = json.loads(line["messages"])
    new = run_new_tokens(MRCR_CHAT_NEW if chat else MRCR_BASE_NEW)
    skipped, n_prompt = False, -1
    tok_path = os.environ.get(ENV_TOKENIZER)
    plain = None if chat else mrcr_plain(messages)
    if tok_path:
        tok = _tokenizer(tok_path)
        n_prompt = _count_messages(messages, tok) if chat else len(tok.encode(plain, add_special_tokens=False))
        skipped = not _fit(n_prompt, new, MRCR_MARGIN)
    spec = {"random_string": line["random_string_to_prepend"], "n_needles": int(line["n_needles"]), "bin_upper": int(line["bin_upper"]), "n_chars": int(line["n_chars"]),
            "desired_msg_index": int(line["desired_msg_index"]), "total_messages": int(line["total_messages"]), "n_tokens": n_prompt, "skipped": skipped}
    if skipped:
        query = SKIP_USER
        if chat:
            spec["chat_messages"] = [{"role": "user", "content": SKIP_USER}]
    elif chat:
        spec["chat_messages"] = messages
        query = messages[-1]["content"]
    else:
        query = plain
    return Doc(task_name=task_name, query=query, choices=[line["answer"]], gold_index=0, specific=spec)


def _mrcr_names():
    return ("score", *[f"score_{n}needle" for n in MRCR_NEEDLES], "prefix_ok", "exact", "skipped")


def _mrcr_score_fn(base: bool):
    def score(doc: Doc, pred: str) -> dict[str, float]:
        spec = doc.specific or {}
        if spec.get("skipped"):
            return {name: NAN for name in _mrcr_names() if name != "skipped"} | {"skipped": 1.0}
        text = pred.lstrip() if base else pred         # base: the model continues "Assistant:" with a space; chat: the text as it is (the official grader does not strip)
        value = mrcr_grade(text, doc.get_golds()[0], spec["random_string"])
        out = {"score": value, "prefix_ok": float(text.startswith(spec["random_string"])), "exact": float(value == 1.0), "skipped": 0.0}
        for n in MRCR_NEEDLES:
            out[f"score_{n}needle"] = _nan_if(spec["n_needles"] == n, value)
        return out

    return score


def mrcr_metric(base: bool) -> SampleLevelMetricGrouping:
    names = _mrcr_names()
    return SampleLevelMetricGrouping(
        metric_name=list(names),
        sample_level_fn=ChatSamples(N_SAMPLES, _mrcr_score_fn(base), names, ks=()),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn={name: (np.mean if name == "skipped" else _members_avg) for name in names},
        higher_is_better={name: name != "skipped" for name in names},
    )


def _mrcr_config(bin_upper: int, chat: bool) -> LightevalTaskConfig:
    def prompt(line, task_name: str = None):
        return mrcr_doc(line, task_name, chat)

    return LightevalTaskConfig(
        name=f"mrcr_{bin_upper}" + ("_chat" if chat else ""),
        prompt_function=prompt,
        hf_repo=prepared(f"mrcr_{bin_upper}"),
        hf_subset="default",
        hf_revision=None,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=MRCR_CHAT_NEW if chat else MRCR_BASE_NEW,
        stop_sequence=[] if chat else MRCR_BASE_STOP,
        metrics=[mrcr_metric(not chat)],
        version=0,
    )


TASKS_TABLE += [_mrcr_config(b, chat) for b in MRCR_BINS for chat in (False, True)]


# ============================================================================================================ FRAMES
FRAMES_CHAT_NEW = 8192
FRAMES_MARGIN = 256
FRAMES_PROMPT_HEAD = "Answer the question below using the Wikipedia articles provided.\n\n"
FRAMES_PROMPT_TAIL = "\n\nQuestion: {question}\n\nGive your final answer on the last line as \"Final answer: <answer>\"."


def frames_articles(articles: list[dict]) -> str:
    return "\n\n".join(f"<article {i + 1}>\nTitle: {a['title']}\nURL: {a['link']}\n\n{a['text'].strip()}\n</article {i + 1}>" for i, a in enumerate(articles))


class _FramesGen(SampleLevelComputation):
    """`answered` (share of non-empty answers) and `truncated` (share of prompts cut in the middle); the accuracy comes from the judge stage."""

    def compute(self, doc, model_response, **kwargs) -> dict[str, float]:
        texts = list(model_response.final_text)
        return {"answered": float(np.mean([1.0 if t and t.strip() else 0.0 for t in texts])) if texts else 0.0,
                "truncated": float(bool((doc.specific or {}).get("truncated")))}


frames_metric = SampleLevelMetricGrouping(
    metric_name=["answered", "truncated"],
    sample_level_fn=_FramesGen(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={"answered": np.mean, "truncated": np.mean},
    higher_is_better={"answered": True, "truncated": False},
)


def frames_chat_prompt(line, task_name: str = None):
    articles = json.loads(line["articles"])
    body = FRAMES_PROMPT_HEAD + frames_articles(articles) + FRAMES_PROMPT_TAIL.format(question=line["question"])
    n_tokens, cut = -1, False
    tok_path = os.environ.get(ENV_TOKENIZER)
    ctx = run_context()
    if tok_path and ctx:
        tok = _tokenizer(tok_path)
        budget = ctx - run_new_tokens(FRAMES_CHAT_NEW) - FRAMES_MARGIN
        if budget <= 0:
            raise ValueError(f"frames_chat: no room for the prompt (context - generation - margin = {budget})")
        body, n_tokens, cut = cut_middle(body, budget, tok)
    reasoning = [t.strip() for t in str(line["reasoning_types"]).split("|") if t.strip()]
    return Doc(
        task_name=task_name,
        query=body,
        choices=[str(line["answer"])],
        gold_index=0,
        specific={"question": line["question"], "gold": str(line["answer"]), "id": int(line["id"]), "reasoning_types": str(line["reasoning_types"]),
                  "reasoning_type": reasoning[0] if reasoning else "unknown", "n_articles": int(line["n_articles"]), "n_tokens": n_tokens, "truncated": cut},
    )


TASKS_TABLE += [
    LightevalTaskConfig(
        name="frames_chat",
        prompt_function=frames_chat_prompt,
        hf_repo=prepared("frames_oracle"),
        hf_subset="default",
        hf_revision=None,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=FRAMES_CHAT_NEW,
        stop_sequence=[],
        metrics=[frames_metric],
        version=0,
    ),
]
