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

import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.utils import as_list


logger = logging.getLogger(__name__)


@dataclass
class CorpusMetricInput:
    pass

    def to_dict(self):
        return asdict(self)


@dataclass
class GenerativeCorpusMetricInput(CorpusMetricInput):
    golds: list[str]
    preds: list[str]


@dataclass
class LogprobCorpusMetricInput(CorpusMetricInput):
    golds: list[int]
    preds: list[float]


@dataclass
class PerplexityCorpusMetricInput(CorpusMetricInput):
    logprobs: list[float]
    weights: list[int]


@dataclass
class COMETCorpusMetricInput(CorpusMetricInput):
    source: str
    hyp: str
    ref: list[str]


class Preparator:
    pass


class GenerativePreparator(Preparator):
    @staticmethod
    def prepare(doc: Doc, model_response: ModelResponse, **kwargs):
        """Prepares an individual generative example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            doc (Doc): The document containing gold references.
            model_response (ModelResponse): The model's response containing predictions.
            **kwargs: Additional keyword arguments.

        Returns:
            GenerativeCorpusMetricInput: Stores the golds and predictions as such
        """
        golds = as_list(doc.get_golds())
        predictions = model_response.final_text
        return GenerativeCorpusMetricInput(golds=golds, preds=predictions)

    def __str__(self):
        attrs = vars(self)
        attr_strs = []
        for k, v in attrs.items():
            if callable(v):
                val_str = v.__name__
            else:
                val_str = str(v)
            attr_strs.append(f"{k}={val_str}")
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"


class COMETPreparator(Preparator):
    @staticmethod
    def prepare(doc: Doc, model_response: ModelResponse, **kwargs) -> "COMETCorpusMetricInput":
        """Prepares a translation example for COMET scoring.

        Reads the source sentence from doc.specific["source_text"] (set by the translation
        template), combines it with the model's hypothesis and the gold reference(s).
        """
        source = (doc.specific or {}).get("source_text", "")
        golds = as_list(doc.get_golds())
        preds = model_response.final_text
        if len(preds) > 1:
            logger.warning("Multiple predictions present, keeping only the first prediction (for COMET).")
        return COMETCorpusMetricInput(source=source, hyp=preds[0], ref=golds)


class LoglikelihoodPreparator(Preparator):
    def __init__(self, is_single_token: bool = False):
        """Init.

        Args:
            is_single_token (bool, optional): True if the preparator is used for single token loglikelihood metrics.
                These metrics are computed faster, as they only compare the single token necessary. Defaults to False.
        """
        self.is_single_token = is_single_token

    def prepare(self, doc: Doc, model_response: ModelResponse, **kwargs) -> LogprobCorpusMetricInput:
        """Prepares an individual loglikelihood example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            doc (Doc): The document containing gold indices and choices.
            model_response (ModelResponse): The model's response containing logprobs.
            **kwargs: Additional keyword arguments.

        Returns:
            LogprobCorpusMetricInput: Stores the golds indices and the model's choice (choice with the highest logprob)
                Only the first gold index is taken for a single token loglikelihood metric
        """
        gold_ixs = as_list(doc.gold_index)
        choices_logprob = model_response.logprobs
        if self.is_single_token:
            if len(gold_ixs) > 1:
                logger.warning(
                    "The current sample has more than one gold available, which is unexpected. We selected only the first one for the corpus aggregation of the loglikelihood metric."
                )
            return LogprobCorpusMetricInput(golds=gold_ixs[0], preds=np.argmax(choices_logprob))

        return LogprobCorpusMetricInput(golds=gold_ixs, preds=np.argmax(choices_logprob))


class TargetPerplexityPreparator(Preparator):
    def __init__(self, units_type: str) -> None:
        """Init.

        Args:
            units_type (str): Basic type of text units we want to use to weight perplexity computations.
                Can be `words` or `bytes`

        Raises:
            ValueError: If the unit type is not words or byte, raises a ValueError
        """
        if units_type not in ["words", "bytes"]:
            raise ValueError("Perplexity must be computed at either the word or byte level.")
        self.units_type = units_type

    def count_units(self, text: str) -> int:
        """Counts the given number of unit in the input text.

        Args:
            text (str): Input text

        Returns:
            int: Number of units of type `self.units_type` in the input text.
        """
        if self.units_type == "words":
            return len(re.split(r"\s+", text))
        if self.units_type == "bytes":
            return len(text.encode("utf-8"))

    def prepare(self, doc: Doc, model_response: ModelResponse, **kwargs):
        """Prepares an individual perplexity example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            doc (Doc): The document containing gold references.
            model_response (ModelResponse): The model's response containing logprobs.
            **kwargs: Additional keyword arguments.

        Returns:
            PerplexityCorpusMetricInput: Stores the measured logprobs and associated text lengths, counted in the reference unit.
        """
        assert len(as_list(doc.gold_index)) == 1, "TargetPerplexityPreparator is meant to be used with a single target reference only."
        gold_ix = as_list(doc.gold_index)[0]
        gold_logprob = model_response.logprobs[gold_ix]
        reference_text_flat = " ".join(doc.get_golds())
        return PerplexityCorpusMetricInput(logprobs=gold_logprob, weights=self.count_units(reference_text_flat))


class PerplexityPreparator(Preparator):
    def __init__(self, units_type: str) -> None:
        """Init.

        Args:
            units_type (str): Basic type of text units we want to use to weight perplexity computations.
                Can be `words` or `bytes`

        Raises:
            ValueError: If the unit type is not words or byte, raises a ValueError
        """
        if units_type not in ["words", "bytes"]:
            raise ValueError("Perplexity must be used with either words or bytes.")
        self.units_type = units_type

    def count_units(self, text: str) -> int:
        """Counts the given number of unit in the input text.

        Args:
            text (str): Input text

        Returns:
            int: Number of units of type `self.units_type` in the input text.
        """
        if self.units_type == "words":
            return len(re.split(r"\s+", text))
        if self.units_type == "bytes":
            return len(text.encode("utf-8"))

    def prepare(self, doc: Doc, model_response: ModelResponse, **kwargs):
        """Prepares an individual perplexity example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            doc (Doc): The document containing gold references.
            model_response (ModelResponse): The model's response containing logprobs.
            **kwargs: Additional keyword arguments.

        Returns:
            PerplexityCorpusMetricInput: Stores the measured logprobs and associated text lengths, counted in the reference unit.
        """
        logprobs_flat = np.sum(model_response.logprobs)

        if doc.original_query is not None:
            reference_text_flat = " ".join([doc.original_query])
        else:
            reference_text_flat = " ".join([doc.query])

        return PerplexityCorpusMetricInput(logprobs=logprobs_flat, weights=self.count_units(reference_text_flat))


def _align_frontier_logprobs(frontier_logprobs: list[dict], reasoning_text: str) -> list[dict]:
    """Trim frontier_logprobs to start at the first token covering reasoning_text.

    GPT-OSS-120b prepends a thinking channel (<|channel|>analysis<|message|>...) before
    the actual response. This finds where reasoning_text starts in the full token byte stream
    and drops all tokens before that point.
    """
    if not frontier_logprobs or not reasoning_text:
        return frontier_logprobs

    target = reasoning_text.encode("utf-8")
    probe = target[:min(32, len(target))]
    if len(probe) < 4:
        return frontier_logprobs

    full_bytes = b"".join(
        bytes(t["bytes"]) if t.get("bytes") else t["token"].encode("utf-8")
        for t in frontier_logprobs
    )
    offset = full_bytes.rfind(probe)
    if offset <= 0:
        return frontier_logprobs  # already aligned or not found

    cursor = 0
    for i, tok in enumerate(frontier_logprobs):
        tok_b = bytes(tok["bytes"]) if tok.get("bytes") else tok["token"].encode("utf-8")
        if cursor + len(tok_b) > offset:
            return frontier_logprobs[i:]
        cursor += len(tok_b)
    return frontier_logprobs


def _build_byte_conf(frontier_logprobs: list[dict], n_bytes: int) -> list[float]:
    """Map each byte of the reasoning span to the frontier token probability covering it.

    Bytes not covered by any token get neutral weight 1.0.
    """
    byte_weights: list[float] = [1.0] * n_bytes
    cursor = 0
    for tok in frontier_logprobs:
        tok_bytes = bytes(tok["bytes"]) if tok.get("bytes") else tok["token"].encode("utf-8")
        prob = math.exp(tok["logprob"])
        for b in range(cursor, min(cursor + len(tok_bytes), n_bytes)):
            byte_weights[b] = prob
        cursor += len(tok_bytes)
        if cursor >= n_bytes:
            break
    return byte_weights


class RBridgePreparator(Preparator):
    """Compute the rBridge weighted NLL score (arXiv:2509.21013).

    Weights each proxy token's NLL by the frontier model's average byte-level
    confidence over the bytes that token covers, then MinMax-normalises.

    Tokenizer resolution order:
      1. Injected via ``set_tokenizer()`` (HF transformers eval — automatic via pipeline)
      2. Loaded from ``TOKENIZER_PATH`` env var (API/litellm eval — same var used by RULER)
      3. None → falls back to unweighted NLL (uniform weights)

    If ``per_token_logprobs`` is unavailable, also falls back to unweighted NLL.
    """

    _tokenizer: Any = None

    @classmethod
    def set_tokenizer(cls, tokenizer: Any) -> None:
        cls._tokenizer = tokenizer

    @classmethod
    def _get_tokenizer(cls) -> Any:
        if cls._tokenizer is not None:
            return cls._tokenizer
        name = os.environ.get("TOKENIZER_PATH")
        if name:
            from transformers import AutoTokenizer
            cls._tokenizer = AutoTokenizer.from_pretrained(name)
            logger.info("RBridgePreparator: loaded tokenizer from TOKENIZER_PATH=%s", name)
        return cls._tokenizer

    def prepare(self, doc: Doc, model_response: ModelResponse, **kwargs) -> PerplexityCorpusMetricInput:
        gold_ix = as_list(doc.gold_index)[0]

        # Fallback: no per-token logprobs → unweighted NLL (same as target_bits_per_byte)
        if not model_response.per_token_logprobs:
            gold_logprob = model_response.logprobs[gold_ix]
            reference_text = doc.choices[gold_ix]
            n_bytes = max(len(reference_text.encode("utf-8")), 1)
            logger.debug("rBridge: per_token_logprobs unavailable, falling back to unweighted NLL/byte")
            return PerplexityCorpusMetricInput(logprobs=gold_logprob, weights=n_bytes)

        tokenizer = self._get_tokenizer()
        # Fallback: no tokenizer → unweighted mean NLL over reasoning tokens
        if tokenizer is None:
            per_token_lps = model_response.per_token_logprobs[gold_ix]
            score = -float(np.mean(per_token_lps)) if per_token_lps else 0.0
            logger.debug("rBridge: tokenizer unavailable, falling back to unweighted mean NLL")
            return PerplexityCorpusMetricInput(logprobs=-score, weights=1)

        specific = doc.specific or {}

        frontier_logprobs = specific["frontier_logprobs"]
        reasoning_byte_start = specific["reasoning_byte_start"]
        reasoning_byte_end = specific["reasoning_byte_end"]
        n_reasoning_bytes = reasoning_byte_end - reasoning_byte_start

        reasoning_text = doc.choices[gold_ix]

        # 1. align logprobs to reasoning_text (drops thinking-channel prefix tokens if present)
        frontier_logprobs = _align_frontier_logprobs(frontier_logprobs, reasoning_text)
        byte_conf = _build_byte_conf(frontier_logprobs, n_reasoning_bytes)

        # 2. tokenize reasoning_text alone — matches how litellm/transformers count choice tokens
        enc = self._tokenizer(
            reasoning_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offset_mapping = enc["offset_mapping"]

        # 3. compute per proxy-token weight
        proxy_weights: list[float] = []
        reasoning_token_pos: list[int] = []
        for token_pos, (cs, ce) in enumerate(offset_mapping):
            tb_start = len(reasoning_text[:cs].encode("utf-8"))
            tb_end = len(reasoning_text[:ce].encode("utf-8"))
            local_s = tb_start
            local_e = min(tb_end, n_reasoning_bytes)
            if local_e <= local_s:
                continue
            w = sum(byte_conf[local_s:local_e]) / (local_e - local_s)
            proxy_weights.append(w)
            reasoning_token_pos.append(token_pos)

        if not proxy_weights:
            return PerplexityCorpusMetricInput(logprobs=0.0, weights=1)

        # 4. MinMax normalize weights
        w_min, w_max = min(proxy_weights), max(proxy_weights)
        if w_max > w_min:
            norm_w = [(w - w_min) / (w_max - w_min) for w in proxy_weights]
        else:
            norm_w = [1.0] * len(proxy_weights)
        if sum(norm_w) == 0:
            norm_w = [1.0] * len(norm_w)

        # 5. per-token proxy NLL
        per_token_lps = model_response.per_token_logprobs[gold_ix]
        nlls = [-lp for lp in per_token_lps]

        # 6. weighted rBridge score
        w_sum = sum(norm_w)
        score = sum(nlls[i] * w for i, w in zip(reasoning_token_pos, norm_w)) / w_sum

        # Return as negative so CorpusLevelPerplexityMetric (which negates) gives the right sign
        return PerplexityCorpusMetricInput(logprobs=-score, weights=1)
