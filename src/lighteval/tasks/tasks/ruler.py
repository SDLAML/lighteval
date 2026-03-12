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

"""
RULER benchmark implementation for lighteval.

RULER (Realistic ULtra-long Language Evaluation with Reasoning) is a long-context
benchmark that requires tokenizer-specific data generation.

Setup:
    Set the RULER_TOKENIZER environment variable to a HuggingFace model name or
    local checkpoint folder containing the tokenizer you want to evaluate.

    export RULER_TOKENIZER=/path/to/model  # or a HF model name like "meta-llama/Llama-3.1-8B"

Data will be generated on the first run and cached under:
    $HF_HOME/lighteval/ruler/<tokenizer_id>/

Subsequent runs (including ablation checkpoints sharing the same tokenizer)
will reuse the cached data instantly.

Task names follow the pattern: ruler_{length}:{subset}
Example: ruler_4096:niah_single_1
"""

import hashlib
import itertools
import logging
import os
import random
import re
import string
import uuid
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBSETS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

DEFAULT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

NUM_SAMPLES = 500
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


def _load_tokenizer(
    tokenizer_path: str,
) -> "PreTrainedTokenizer | PreTrainedTokenizerFast":
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _get_model_arch(tokenizer_path: str) -> str:
    """Read the first architecture name from config.json in a local checkpoint.

    Returns the architecture string (e.g. "LlamaForCausalLM") for use as a
    stable, human-readable cache directory name that is shared across all
    checkpoint steps of the same model.
    Falls back to a sanitised path hash when config.json is absent (e.g. HF
    hub model IDs passed as strings without a local directory).
    """
    import json

    config_path = os.path.join(tokenizer_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            archs = cfg.get("architectures", [])
            if archs:
                return archs[0]
        except Exception:
            pass

    # Fallback for HF hub names or unreadable configs
    h = hashlib.md5(tokenizer_path.encode()).hexdigest()[:8]
    safe = re.sub(r"[/\\]", "_", tokenizer_path)[-40:]
    return f"{safe}_{h}"


def _get_cache_dir(tokenizer_path: str) -> Path:
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    arch = _get_model_arch(tokenizer_path)
    return Path(hf_home) / "lighteval" / "ruler" / arch


# ---------------------------------------------------------------------------
# NIAH (Needle-In-A-Haystack) generation
# Ported from lm-evaluation-harness lm_eval/tasks/ruler/prepare_niah.py
# Original: Copyright (c) 2024, NVIDIA CORPORATION. Apache 2.0 License.
# ---------------------------------------------------------------------------

NIAH_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
)

NIAH_DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))


def _ensure_nltk():
    import nltk
    from packaging.version import parse as parse_version
    from importlib.metadata import version

    NLTK_MIN_VERSION = "3.9.1"
    nltk_version = parse_version(version("nltk"))
    assert nltk_version >= parse_version(NLTK_MIN_VERSION), (
        f"nltk >= {NLTK_MIN_VERSION} required (got {nltk_version}). "
        "Please update: pip install --upgrade nltk"
    )
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")


@lru_cache(maxsize=1)
def _get_niah_words():
    import wonderwords

    r = wonderwords.RandomWord()
    nouns = r._categories["nouns"]
    adjs = r._categories["adjectives"]
    words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
    return sorted(list(set(words)))


@lru_cache(maxsize=1)
def _get_essay_haystack() -> list[str]:
    import datasets as hf_datasets

    essay = hf_datasets.load_dataset("baber/paul_graham_essays", split="train")["text"]
    text = " ".join(essay)
    text = re.sub(r"\s+", " ", text)
    return text.split(" ")


@lru_cache(maxsize=32)
def _cached_sent_tokenize(text: str) -> list[str]:
    from nltk import sent_tokenize

    return sent_tokenize(text)


def _generate_random_niah(type_needle: str) -> str:
    if type_needle == "numbers":
        lower = 10**6
        upper = 10**7 - 1
        return str(random.randint(lower, upper))
    elif type_needle == "words":
        return random.choice(_get_niah_words())
    elif type_needle == "uuids":
        return str(uuid.UUID(int=random.getrandbits(128), version=4))
    raise NotImplementedError(f"Unknown needle type: {type_needle}")


def _niah_generate_input_output(
    num_haystack: int,
    haystack,
    *,
    type_haystack: str,
    num_needle_k: int,
    type_needle_k: str,
    num_needle_v: int,
    type_needle_v: str,
    template: str,
    num_needle_q: int = 1,
    random_seed: int = RANDOM_SEED,
) -> tuple[str, list[str], str]:
    needle = NIAH_NEEDLE
    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        keys.append(_generate_random_niah(type_needle_k))
        value = []
        for _ in range(num_needle_v):
            value.append(_generate_random_niah(type_needle_v))
            needles.append(
                needle.format(
                    type_needle_v=type_needle_v,
                    key=keys[-1],
                    value=value[-1],
                )
            )
        values.append(value)

    random.Random(random_seed).shuffle(needles)

    if type_haystack == "essay":
        assert isinstance(haystack, list)
        text = " ".join(haystack[:num_haystack])
        document_sents = _cached_sent_tokenize(text.strip())
        insertion_positions = (
            [0]
            + sorted(
                [
                    int(len(document_sents) * (depth / 100))
                    for depth in random.sample(NIAH_DEPTHS, len(needles))
                ]
            )
            + [len(document_sents)]
        )
        document_sents_list = []
        for i in range(1, len(insertion_positions)):
            last_pos = insertion_positions[i - 1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i - 1 < len(needles):
                document_sents_list.append(needles[i - 1])
        context = " ".join(document_sents_list)
    elif type_haystack == "repeat":
        sentences = [haystack] * num_haystack
        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)
    elif type_haystack == "needle":
        sentences = [
            haystack.format(
                type_needle_v=type_needle_v,
                key=_generate_random_niah(type_needle_k),
                value=_generate_random_niah(type_needle_v),
            )
            for _ in range(num_haystack)
        ]
        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)
    else:
        raise NotImplementedError(f"Unknown haystack type: {type_haystack}")

    indices = random.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = (
        ", ".join(queries[:-1]) + ", and " + queries[-1]
        if len(queries) > 1
        else queries[0]
    )

    tmpl = template
    tnv = type_needle_v
    if num_needle_q * num_needle_v == 1:
        tmpl = tmpl.replace("Some", "A")
        tmpl = tmpl.replace("are all", "is")
        tmpl = tmpl.replace("are", "is")
        tmpl = tmpl.replace("answers", "answer")
        tnv = tnv[:-1]  # remove trailing 's'

    input_text = tmpl.format(type_needle_v=tnv, context=context, query=query)
    return input_text, answers, query


def _niah_generate_samples(
    haystack,
    tokenizer,
    *,
    max_seq_length: int,
    type_haystack: str,
    type_needle_k: str,
    type_needle_v: str,
    template: str,
    num_samples: int = NUM_SAMPLES,
    tokens_to_generate: int = 128,
    num_needle_v: int = 1,
    num_needle_k: int = 1,
    num_needle_q: int = 1,
    random_seed: int = RANDOM_SEED,
) -> list[dict]:
    budget = max_seq_length
    num_needle_k = max(num_needle_k, num_needle_q)

    if type_haystack == "essay":
        incremental = 500
    elif type_haystack in ("repeat", "needle"):
        incremental = 25 if max_seq_length >= 4096 else 5
    else:
        incremental = 25

    num_haystack = incremental
    total_tokens = 0

    def _gen_prefix(tnv, nq, nv, query):
        if nq * nv == 1:
            return f"The special magic {tnv[:-1]} for {query} mentioned in the provided text is"
        return f"The special magic {tnv} for {query} mentioned in the provided text are"

    tnv_base = type_needle_v

    while total_tokens + tokens_to_generate < budget:
        input_text, _, query = _niah_generate_input_output(
            num_haystack,
            haystack,
            type_haystack=type_haystack,
            num_needle_k=num_needle_k,
            type_needle_k=type_needle_k,
            num_needle_v=num_needle_v,
            type_needle_v=type_needle_v,
            template=template,
            num_needle_q=num_needle_q,
            random_seed=random_seed,
        )
        gen_prefix = _gen_prefix(tnv_base, num_needle_q, num_needle_v, query)
        prompt = input_text + " " + gen_prefix
        total_tokens = len(tokenizer(prompt).input_ids)
        if total_tokens + tokens_to_generate > budget:
            num_haystack -= incremental
            break
        if type_haystack == "essay" and num_haystack > len(haystack):
            num_haystack = len(haystack)
            break
        num_haystack += incremental

    # Clamp to minimum: the sizing loop can decrement to 0 for very small budgets
    # (e.g. 256 tokens: 0+128<256 → enter loop → 5 elements already exceed → num_haystack=0)
    num_haystack = max(incremental, num_haystack)

    write_jsons = []
    for index in tqdm(
        range(num_samples), desc=f"Generating NIAH samples | {max_seq_length}"
    ):
        used_haystack = num_haystack
        input_text = answer = query = gen_prefix = length = None
        while True:
            try:
                input_text, answer, query = _niah_generate_input_output(
                    used_haystack,
                    haystack,
                    type_haystack=type_haystack,
                    num_needle_k=num_needle_k,
                    type_needle_k=type_needle_k,
                    num_needle_v=num_needle_v,
                    type_needle_v=type_needle_v,
                    template=template,
                    num_needle_q=num_needle_q,
                    random_seed=random_seed,
                )
                gen_prefix = _gen_prefix(tnv_base, num_needle_q, num_needle_v, query)
                prompt = input_text + " " + gen_prefix
                length = len(tokenizer(prompt).input_ids) + tokens_to_generate
                assert length <= budget
                break
            except Exception:
                if used_haystack > incremental:
                    used_haystack -= incremental
                else:
                    break

        if answer is None:
            continue

        write_jsons.append(
            {
                "index": index,
                "input": input_text,
                "outputs": answer,
                "length": length,
                "max_length": max_seq_length,
                "gen_prefix": gen_prefix,
            }
        )

    return write_jsons


# ---------------------------------------------------------------------------
# VT (Variable Tracking) generation
# Ported from lm-evaluation-harness lm_eval/tasks/ruler/vt_utils.py
# Original: Copyright (c) 2024, NVIDIA CORPORATION. Apache 2.0 License.
# ---------------------------------------------------------------------------

VT_CONFIG = {
    "tokens_to_generate": 30,
    "template": (
        "Memorize and track the chain(s) of variable assignment hidden in the "
        "following text.\n\n{context}\n"
        "Question: Find all variables that are assigned the value {query} in the text above."
    ),
    "answer_prefix": (
        " Answer: According to the chain(s) of variable assignment in the text above, "
        "{num_v} variables are assgined the value {query}, they are: "
    ),
}
VT_TEMPLATE = VT_CONFIG["template"] + VT_CONFIG["answer_prefix"]


def _vt_generate_chains(
    num_chains: int, num_hops: int, is_icl: bool = False
) -> tuple[list[list[str]], list[list[str]]]:
    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    vars_all = [
        "".join(random.choices(string.ascii_uppercase, k=k)).upper()
        for _ in range((num_hops + 1) * num_chains)
    ]
    while len(set(vars_all)) < num_chains * (num_hops + 1):
        vars_all.append("".join(random.choices(string.ascii_uppercase, k=k)).upper())

    vars_ret = []
    chains_ret = []
    for i in range(0, len(vars_all), num_hops + 1):
        this_vars = vars_all[i : i + num_hops + 1]
        vars_ret.append(this_vars)
        this_chain = [f"VAR {this_vars[0]} = {np.random.randint(10000, 99999)}"]
        for j in range(num_hops):
            this_chain.append(f"VAR {this_vars[j + 1]} = VAR {this_vars[j]} ")
        chains_ret.append(this_chain)
    return vars_ret, chains_ret


def _vt_generate_input_output(num_noises, num_chains, num_hops, is_icl=False):
    vars_list, chains = _vt_generate_chains(num_chains, num_hops, is_icl=is_icl)
    noise = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"
    sentences = [noise] * num_noises
    if len(sentences) <= len(chains[0]):
        sentences = [
            n + "." if len(n.strip()) > 0 else n
            for n in [x for noise in sentences for x in noise.split(".")]
        ]
        try:
            assert len(sentences) > len(chains[0])
        except AssertionError:
            chains = [chain[: len(sentences) - 1] for chain in chains]
    for chain_i in chains:
        positions = list(sorted(random.sample(range(len(sentences)), len(chain_i))))
        for insert_pi, j in zip(positions, range(len(chain_i))):
            sentences.insert(insert_pi + j, chain_i[j])

    context = " ".join(sentences)
    context = context.replace(". \n", ".\n")
    value = chains[0][0].split("=")[-1].strip()
    input_text = VT_TEMPLATE.format(context=context, query=value, num_v=num_hops + 1)
    return input_text, vars_list[0]


def _vt_randomize_icl(icl_example: str) -> str:
    icl_tgt_cut = icl_example.index(VT_CONFIG["answer_prefix"][-10:])
    icl_tgt = icl_example[icl_tgt_cut + 10 :].strip().split()
    for item in icl_tgt:
        new_item = "".join(random.choices(string.ascii_uppercase, k=len(item))).upper()
        icl_example = icl_example.replace(item, new_item)
    return icl_example


def _vt_generate_samples(
    tokenizer,
    max_seq_length: int,
    num_samples: int = NUM_SAMPLES,
    incremental: int = 10,
    num_chains: int = 1,
    num_hops: int = 4,
    tokens_to_generate: int = 30,
) -> list[dict]:
    budget = max_seq_length

    # --- Step 1: Generate ICL example within a 500-token budget (matches reference) ---
    # Reference: get_dataset() calls sys_vartrack_w_noise_random(max_seq_length=500, incremental=5)
    icl_budget = 500
    icl_incremental = 5
    icl_num_noises = icl_incremental
    icl_total_tokens = 0
    while icl_total_tokens + tokens_to_generate < icl_budget:
        icl_text, _ = _vt_generate_input_output(
            icl_num_noises, num_chains, num_hops, is_icl=True
        )
        icl_total_tokens = len(tokenizer(icl_text).input_ids)
        if icl_total_tokens + tokens_to_generate > icl_budget:
            icl_num_noises -= icl_incremental
            break
        icl_num_noises += icl_incremental

    icl_num_noises = max(icl_incremental, icl_num_noises)
    icl_text_raw, icl_answer = _vt_generate_input_output(
        icl_num_noises, num_chains, num_hops, is_icl=True
    )
    # icl_text_raw includes the answer_prefix (VT_TEMPLATE = template + answer_prefix)
    icl_out = " ".join(icl_answer)
    icl_str = (
        icl_text_raw + " " + icl_out
    )  # full ICL text: question + gen_prefix + answers
    example_tokens = len(tokenizer(icl_str + "\n\n").input_ids)

    # --- Step 2: Find num_noises for main examples ---
    num_noises = incremental
    total_tokens = 0
    while total_tokens + tokens_to_generate + example_tokens < budget:
        input_text, _ = _vt_generate_input_output(num_noises, num_chains, num_hops)
        total_tokens = len(tokenizer(input_text).input_ids)
        if total_tokens + tokens_to_generate + example_tokens > budget:
            num_noises -= incremental
            break
        num_noises += incremental

    num_noises = max(incremental, num_noises)

    write_jsons = []
    for index in tqdm(
        range(num_samples), desc=f"Generating VT samples | {max_seq_length}"
    ):
        used_noises = num_noises
        input_text = answer = length = None
        while True:
            try:
                input_text, answer = _vt_generate_input_output(
                    used_noises, num_chains, num_hops
                )
                length = (
                    len(tokenizer(input_text).input_ids)
                    + tokens_to_generate
                    + example_tokens
                )
                assert length <= budget
                break
            except Exception:
                if used_noises > incremental:
                    used_noises -= incremental
                else:
                    break

        if answer is None:
            continue

        # Insert ICL example between any model template prefix and the task template
        cutoff = input_text.index(VT_CONFIG["template"][:20])
        input_text = (
            input_text[:cutoff]
            + _vt_randomize_icl(icl_str)
            + "\n\n"
            + input_text[cutoff:]
        )

        # Split off gen_prefix (the answer_prefix at the end of VT_TEMPLATE)
        gen_prefix_index = input_text.rfind(
            " Answer: According to the chain(s) of variable assignment"
        )
        gen_prefix = input_text[gen_prefix_index:].strip()
        input_text = input_text[:gen_prefix_index]

        write_jsons.append(
            {
                "index": index,
                "input": input_text,
                "outputs": answer,
                "length": length,
                "max_length": max_seq_length,
                "gen_prefix": gen_prefix,
            }
        )

    return write_jsons


# ---------------------------------------------------------------------------
# CWE (Common Word Extraction) generation
# Ported from lm-evaluation-harness lm_eval/tasks/ruler/cwe_utils.py
# Original: Copyright (c) 2024, NVIDIA CORPORATION. Apache 2.0 License.
# ---------------------------------------------------------------------------

CWE_CONFIG = {
    "tokens_to_generate": 120,
    "template": (
        "Below is a numbered list of words. In these words, some appear more often than others. "
        "Memorize the ones that appear most often.\n{context}\n"
        "Question: What are the 10 most common words in the above list?"
    ),
    "answer_prefix": " Answer: The top 10 words that appear most often in the list are:",
}
CWE_TEMPLATE = CWE_CONFIG["template"] + CWE_CONFIG["answer_prefix"]
_CWE_RNG = random.Random(RANDOM_SEED)


@lru_cache(maxsize=1)
def _get_cwe_words() -> list[str]:
    import wonderwords

    r = wonderwords.RandomWord()
    words = sorted(
        list(
            set(
                [
                    item
                    for x in ["noun", "adjective", "verb"]
                    for item in r._categories[x]
                ]
            )
        )
    )
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(words)
    return words


def _cwe_get_example(
    num_words: int,
    words: list[str],
    common_repeats: int = 30,
    uncommon_repeats: int = 3,
    common_nums: int = 10,
):
    word_list_full = random.sample(words, num_words)
    common, uncommon = word_list_full[:common_nums], word_list_full[common_nums:]
    word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
    _CWE_RNG.shuffle(word_list)
    context = " ".join([f"{i + 1}. {w}" for i, w in enumerate(word_list)])
    return context, common


def _cwe_generate_input_output(num_words: int, max_seq_length: int, words: list[str]):
    if max_seq_length < 4096:
        ctx_ex, ans_ex = _cwe_get_example(20, words, 3, 1, 10)
        context, answer = _cwe_get_example(num_words, words, 6, 1, 10)
    else:
        ctx_ex, ans_ex = _cwe_get_example(40, words, 10, 3, 10)
        context, answer = _cwe_get_example(num_words, words, 30, 3, 10)

    input_example = CWE_TEMPLATE.format(context=ctx_ex, query="") + " ".join(
        [f"{i + 1}. {w}" for i, w in enumerate(ans_ex)]
    )
    input_text = CWE_TEMPLATE.format(context=context, query="")
    return input_example, input_text, answer


def _cwe_generate_samples(
    tokenizer,
    max_seq_length: int,
    num_samples: int = NUM_SAMPLES,
    incremental: int = 10,
    tokens_to_generate: int = 120,
) -> list[dict]:
    words = _get_cwe_words()
    budget = max_seq_length

    num_words = incremental
    total_tokens = 0
    while total_tokens + tokens_to_generate < budget:
        input_example, input_text, answer = _cwe_generate_input_output(
            num_words, max_seq_length, words
        )
        total_tokens = len(
            tokenizer(
                input_example
                + "\n"
                + input_text
                + " "
                + " ".join(f"{i+1}. {w}" for i, w in enumerate(answer))
            ).input_ids
        )
        if total_tokens + tokens_to_generate > budget:
            num_words -= incremental
            break
        num_words += incremental
        if num_words > len(words):
            num_words = len(words)
            break

    num_words = max(incremental, num_words)

    write_jsons = []
    for index in tqdm(
        range(num_samples), desc=f"Generating CWE samples | {max_seq_length}"
    ):
        used_words = num_words
        input_example = input_text = answer = length = None
        while True:
            try:
                input_example, input_text, answer = _cwe_generate_input_output(
                    used_words, max_seq_length, words
                )
                length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                assert length <= budget
                break
            except Exception:
                if used_words > incremental:
                    used_words -= incremental
                else:
                    break

        if answer is None:
            continue

        gen_prefix_idx = input_text.rfind(CWE_CONFIG["answer_prefix"])
        gen_prefix = input_text[gen_prefix_idx:].strip()
        input_text = input_text[:gen_prefix_idx]
        write_jsons.append(
            {
                "index": index,
                "input": input_text.strip(),
                "outputs": answer,
                "length": length,
                "max_length": max_seq_length,
                "gen_prefix": gen_prefix,
            }
        )

    return write_jsons


# ---------------------------------------------------------------------------
# FWE (Frequent Word Extraction) generation
# Ported from lm-evaluation-harness lm_eval/tasks/ruler/fwe_utils.py
# Original: Copyright (c) 2024, NVIDIA CORPORATION. Apache 2.0 License.
# ---------------------------------------------------------------------------

FWE_CONFIG = {
    "tokens_to_generate": 50,
    "template": (
        "Read the following coded text and track the frequency of each coded word. "
        "Find the three most frequently appeared coded words. {context}\n"
        "Question: Do not provide any explanation. Please ignore the dots '....'. "
        "What are the three most frequently appeared words in the above coded text?"
    ),
    "answer_prefix": " Answer: According to the coded text above, the three most frequently appeared words are:",
}
FWE_TEMPLATE = FWE_CONFIG["template"] + FWE_CONFIG["answer_prefix"]
FWE_SEED = RANDOM_SEED


def _fwe_generate_input_output(
    max_len: int,
    tokenizer,
    num_words: int = -1,
    coded_wordlen: int = 6,
    vocab_size: int = 2000,
    incremental: int = 10,
    alpha: float = 2.0,
) -> tuple[str, list[str], int]:
    from scipy.special import zeta

    vocab = [
        "".join(random.choices(string.ascii_lowercase, k=coded_wordlen))
        for _ in range(vocab_size)
    ]
    while len(set(vocab)) < vocab_size:
        vocab.append("".join(random.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(list(set(vocab)))
    random.Random(FWE_SEED).shuffle(vocab)
    vocab[0] = "..."

    def gen_text(nw):
        k = np.arange(1, len(vocab) + 1)
        sampled_cnt = nw * (k**-alpha) / zeta(alpha)
        sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))]
        sampled_words = [x for wlst in sampled_words for x in wlst]
        random.Random(FWE_SEED).shuffle(sampled_words)
        return (
            FWE_TEMPLATE.format(context=" ".join(sampled_words), query=""),
            vocab[1:4],
        )

    if num_words > 0:
        text, answer = gen_text(num_words)
        while len(tokenizer(text).input_ids) > max_len:
            num_words -= incremental
            text, answer = gen_text(num_words)
    else:
        num_words = max_len // coded_wordlen
        text, answer = gen_text(num_words)
        while len(tokenizer(text).input_ids) < max_len:
            num_words += incremental
            text, answer = gen_text(num_words)
        num_words -= incremental
    text, answer = gen_text(num_words)
    return text, answer, num_words


def _fwe_generate_samples(
    tokenizer,
    max_seq_length: int,
    num_samples: int = NUM_SAMPLES,
    vocab_size: int = -1,
    coded_wordlen: int = 6,
    alpha: float = 2.0,
    tokens_to_generate: int = 50,
) -> list[dict]:
    budget = max_seq_length
    input_max_len = budget - tokens_to_generate
    vs = max_seq_length // 50 if vocab_size == -1 else vocab_size

    _, _, num_example_words = _fwe_generate_input_output(
        input_max_len,
        tokenizer,
        coded_wordlen=coded_wordlen,
        vocab_size=vs,
        incremental=input_max_len // 32,
        alpha=alpha,
    )

    write_jsons = []
    for index in tqdm(
        range(num_samples), desc=f"Generating FWE samples | {max_seq_length}"
    ):
        input_text, answer, _ = _fwe_generate_input_output(
            input_max_len,
            tokenizer,
            num_words=num_example_words,
            coded_wordlen=coded_wordlen,
            vocab_size=vs,
            incremental=input_max_len // 32,
            alpha=alpha,
        )
        length = len(tokenizer(input_text).input_ids) + tokens_to_generate
        assert length <= budget

        # Strip answer prefix from input
        ans_prefix_idx = input_text.rfind(FWE_CONFIG["answer_prefix"])
        gen_prefix = input_text[ans_prefix_idx:].strip()
        input_text_clean = input_text[:ans_prefix_idx].strip()

        write_jsons.append(
            {
                "index": index,
                "input": input_text_clean,
                "outputs": answer,
                "length": length,
                "max_length": max_seq_length,
                "gen_prefix": gen_prefix,
            }
        )

    return write_jsons


# ---------------------------------------------------------------------------
# QA generation (SQuAD / HotpotQA)
# Ported from lm-evaluation-harness lm_eval/tasks/ruler/qa_utils.py
# Original: Copyright (c) 2024, NVIDIA CORPORATION. Apache 2.0 License.
# ---------------------------------------------------------------------------

QA_CONFIG = {
    "tokens_to_generate": 32,
    "template": (
        "Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given documents.\n\n{context}\n\n"
        "Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\n"
        "Question: {query}"
    ),
    "answer_prefix": "Answer:",
}
QA_DOCUMENT_PROMPT = "Document {i}:\n{document}"


def _qa_download_json(url: str) -> dict:
    import hashlib, json, requests  # noqa: E401

    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_path = Path(hf_home) / "lighteval" / "ruler_qa_cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    h = hashlib.md5(url.encode()).hexdigest()
    file_path = cache_path / f"{h}.json"
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)

    logger.info(f"Downloading {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    with open(file_path, "w") as f:
        json.dump(data, f)
    return data


@lru_cache(maxsize=1)
def _read_squad() -> tuple[list[dict], list[str]]:
    data = _qa_download_json(
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    )
    total_docs = [p["context"] for d in data["data"] for p in d["paragraphs"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data["data"]:
        more_docs = [total_docs_dict[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            for qas in p["qas"]:
                if not qas["is_impossible"]:
                    total_qas.append(
                        {
                            "query": qas["question"],
                            "outputs": [a["text"] for a in qas["answers"]],
                            "context": [total_docs_dict[p["context"]]],
                            "more_context": [
                                idx
                                for idx in more_docs
                                if idx != total_docs_dict[p["context"]]
                            ],
                        }
                    )
    return total_qas, total_docs


@lru_cache(maxsize=1)
def _read_hotpotqa() -> tuple[list[dict], list[str]]:
    data = _qa_download_json(
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    )
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d["context"]]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data:
        total_qas.append(
            {
                "query": d["question"],
                "outputs": [d["answer"]],
                "context": [
                    total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d["context"]
                ],
            }
        )
    return total_qas, total_docs


def _qa_generate_input_output(
    index: int, num_docs: int, qas: list[dict], docs: list[str]
) -> tuple[str, list[str]]:
    curr_q = qas[index]["query"]
    curr_a = qas[index]["outputs"]
    curr_docs = qas[index]["context"]
    curr_more = qas[index].get("more_context", [])

    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [
                i for i, _ in enumerate(docs) if i not in curr_docs + curr_more
            ]
            all_docs = (
                curr_docs
                + curr_more
                + random.sample(
                    addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
                )
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [docs[idx] for idx in all_docs]
    else:
        all_docs = docs

    random.Random(RANDOM_SEED).shuffle(all_docs)
    context = "\n\n".join(
        [QA_DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)]
    )
    input_text = QA_CONFIG["template"].format(context=context, query=curr_q)
    return input_text, curr_a


def _qa_generate_samples(
    tokenizer,
    docs: list[str],
    qas: list[dict],
    max_seq_length: int,
    num_samples: int = NUM_SAMPLES,
    tokens_to_generate: int = 32,
    incremental: int = 10,
) -> list[dict]:
    budget = max_seq_length
    gen_prefix = QA_CONFIG["answer_prefix"]

    num_docs = incremental
    total_tokens = 0
    while total_tokens + tokens_to_generate < budget:
        input_text, _ = _qa_generate_input_output(0, num_docs, qas=qas, docs=docs)
        prompt = input_text + " " + gen_prefix
        total_tokens = len(tokenizer(prompt).input_ids)
        if total_tokens + tokens_to_generate > budget:
            num_docs -= incremental
            break
        num_docs += incremental
        if num_docs > len(docs):
            num_docs = len(docs)
            break

    num_docs = max(incremental, num_docs)

    write_jsons = []
    for index in tqdm(
        range(num_samples), desc=f"Generating QA samples | {max_seq_length}"
    ):
        used_docs = num_docs
        input_text = answer = length = None
        while True:
            try:
                input_text, answer = _qa_generate_input_output(
                    index, used_docs, qas=qas, docs=docs
                )
                prompt = input_text + " " + gen_prefix
                length = len(tokenizer(prompt).input_ids) + tokens_to_generate
                assert length <= budget
                break
            except Exception:
                if used_docs > incremental:
                    used_docs -= incremental
                else:
                    break

        if answer is None:
            continue

        write_jsons.append(
            {
                "index": index,
                "input": input_text,
                "outputs": answer,
                "length": length,
                "max_length": max_seq_length,
                "gen_prefix": gen_prefix,
            }
        )

    return write_jsons


# ---------------------------------------------------------------------------
# Per-subset generation dispatcher
# ---------------------------------------------------------------------------


def _generate_subset(subset: str, length: int, tokenizer) -> list[dict]:
    if subset == "niah_single_1":
        return _niah_generate_samples(
            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.",
            tokenizer,
            max_seq_length=length,
            type_haystack="repeat",
            type_needle_k="words",
            type_needle_v="numbers",
            template=NIAH_TEMPLATE,
        )
    elif subset == "niah_single_2":
        _ensure_nltk()
        return _niah_generate_samples(
            _get_essay_haystack(),
            tokenizer,
            max_seq_length=length,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            template=NIAH_TEMPLATE,
        )
    elif subset == "niah_single_3":
        _ensure_nltk()
        return _niah_generate_samples(
            _get_essay_haystack(),
            tokenizer,
            max_seq_length=length,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="uuids",
            template=NIAH_TEMPLATE,
        )
    elif subset == "niah_multikey_1":
        _ensure_nltk()
        return _niah_generate_samples(
            _get_essay_haystack(),
            tokenizer,
            max_seq_length=length,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_k=4,
            template=NIAH_TEMPLATE,
        )
    elif subset == "niah_multikey_2":
        return _niah_generate_samples(
            NIAH_NEEDLE,
            tokenizer,
            max_seq_length=length,
            type_haystack="needle",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_k=4,
            template=NIAH_TEMPLATE,
        )
    elif subset == "niah_multikey_3":
        return _niah_generate_samples(
            NIAH_NEEDLE,
            tokenizer,
            max_seq_length=length,
            type_haystack="needle",
            type_needle_k="uuids",
            type_needle_v="uuids",
            num_needle_k=4,
            template=NIAH_TEMPLATE,
        )
    elif subset == "niah_multiquery":
        _ensure_nltk()
        return _niah_generate_samples(
            _get_essay_haystack(),
            tokenizer,
            max_seq_length=length,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_q=4,
            template=NIAH_TEMPLATE,
        )
    elif subset == "niah_multivalue":
        _ensure_nltk()
        return _niah_generate_samples(
            _get_essay_haystack(),
            tokenizer,
            max_seq_length=length,
            type_haystack="essay",
            type_needle_k="words",
            type_needle_v="numbers",
            num_needle_v=4,
            template=NIAH_TEMPLATE,
        )
    elif subset == "vt":
        return _vt_generate_samples(tokenizer, max_seq_length=length)
    elif subset == "cwe":
        return _cwe_generate_samples(tokenizer, max_seq_length=length)
    elif subset == "fwe":
        return _fwe_generate_samples(tokenizer, max_seq_length=length)
    elif subset == "qa_1":
        qas, docs = _read_squad()
        return _qa_generate_samples(
            tokenizer, docs=docs, qas=qas, max_seq_length=length
        )
    elif subset == "qa_2":
        qas, docs = _read_hotpotqa()
        return _qa_generate_samples(
            tokenizer, docs=docs, qas=qas, max_seq_length=length
        )
    else:
        raise ValueError(f"Unknown RULER subset: {subset}")


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def ensure_ruler_cache(
    tokenizer_name: str,
    lengths: list[int] | None = None,
    subsets: list[str] | None = None,
) -> Path:
    """Generate and cache RULER data for the given tokenizer.

    Data is stored at $HF_HOME/lighteval/ruler/<tokenizer_id>/<length>/<subset>/.
    Each subset is saved independently so evaluating one subset does not trigger
    generation of all subsets.  Already-generated (length, subset) pairs are skipped.

    Returns the cache base directory.
    """
    lengths = lengths or DEFAULT_LENGTHS
    subsets = subsets or SUBSETS
    cache_base = _get_cache_dir(tokenizer_name)
    cache_base.mkdir(parents=True, exist_ok=True)

    missing = [
        (l, s)
        for l in lengths
        for s in subsets
        if not (cache_base / str(l) / s / "dataset_dict.json").exists()
    ]
    if not missing:
        return cache_base

    logger.info(
        f"Loading tokenizer from {tokenizer_name} for RULER data generation ..."
    )
    tokenizer = _load_tokenizer(tokenizer_name)

    for length, subset in missing:
        logger.info(f"Generating RULER data: length={length} subset={subset} ...")
        try:
            samples = _generate_subset(subset, length, tokenizer)
            if not samples:
                raise ValueError("no samples generated (context length too small?)")
            subset_dir = cache_base / str(length) / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            DatasetDict({"test": Dataset.from_list(samples)}).save_to_disk(
                str(subset_dir)
            )
            logger.info(f"Saved {len(samples)} samples to {subset_dir}")
        except Exception as e:
            logger.warning(f"Skipping subset {subset} at length {length}: {e}")

    return cache_base


# ---------------------------------------------------------------------------
# Prompt function
# ---------------------------------------------------------------------------


def ruler_prompt(line: dict, task_name: str = None) -> Doc:
    outputs = line["outputs"]
    # Append gen_prefix so the model receives the completion-eliciting prefix
    # (matches lm-eval-harness YAML: doc_to_text="{{input}}" + gen_prefix="{{gen_prefix}}")
    query = line["input"]
    gp = line.get("gen_prefix", "")
    if gp:
        query = query + " " + gp
    return Doc(
        query=query,
        choices=outputs,
        gold_index=list(range(len(outputs))),
        task_name=task_name,
    )


# ---------------------------------------------------------------------------
# Task config builder
# ---------------------------------------------------------------------------


def get_ruler_tasks(
    tokenizer_name: str,
    lengths: list[int] | None = None,
    subsets: list[str] | None = None,
) -> list[LightevalTaskConfig]:
    """Register RULER task configs for all lengths.

    Data is generated on demand in download_dataset_worker (keyed off
    RULER_TOKENIZER) so only the lengths actually evaluated are generated.
    """
    lengths = lengths or DEFAULT_LENGTHS
    subsets = subsets or SUBSETS

    cache_base = _get_cache_dir(tokenizer_name)

    tasks = []
    for subset in subsets:
        for length in lengths:
            # Each (length, subset) pair has its own directory so loading one
            # subset never triggers generation of all subsets.
            cache_path = cache_base / str(length) / subset
            metric = [Metrics.ruler_match]
            gen_size = (
                128
                if "niah" in subset
                else (
                    30 if subset == "vt" else 120 if subset == "cwe" else 50
                )  # fwe and qa
            )
            tasks.append(
                LightevalTaskConfig(
                    name=f"ruler_{length}:{subset}",
                    prompt_function=ruler_prompt,
                    hf_repo=str(cache_path),  # local path → load_from_disk
                    hf_subset="default",
                    hf_avail_splits=["test"],
                    evaluation_splits=["test"],
                    few_shots_split=None,
                    few_shots_select=None,
                    generation_size=gen_size,
                    metrics=metric,
                    stop_sequence=None,
                    version=0,
                )
            )

    return tasks


# ---------------------------------------------------------------------------
# TASKS_TABLE — populated when RULER_TOKENIZER env var is set
# ---------------------------------------------------------------------------

_ruler_tokenizer = os.environ.get("RULER_TOKENIZER")

if _ruler_tokenizer:
    TASKS_TABLE = get_ruler_tasks(_ruler_tokenizer)
else:
    TASKS_TABLE = []
    logger.debug(
        "RULER_TOKENIZER env var not set — RULER tasks not loaded. "
        "Set RULER_TOKENIZER=<hf_model_or_path> before importing this module."
    )
