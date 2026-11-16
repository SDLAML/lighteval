# MIT License
#
# titan_eval (owner 11-16): the EXACT prompt size of a long-context chat item, for the answer-budget rule of `over_context: zero` (answer_fit.py).
#
# OLMES (oe_eval/models/eleuther_vllm_causallms.py:213) computes the answer budget from the tokenized prompt itself: min(task budget, max_length - prompt tokens), no margin; HELMET's own buffer is 0
# (helmet.py:386). So does this module: it renders a doc exactly as the vLLM model of the fork does (PromptManager: the model's chat template through its tokenizer, the same messages, `add_generation_prompt`,
# the `chat_template_kwargs` that carry the thinking / non-thinking switch, the system prompt, the named template file, a doc's own `chat_messages`, a `plain_text` doc as it is) and counts the tokens of the
# rendered string with the model's tokenizer and the model's `add_special_tokens` (False for chat: nothing is added, no BOS). The vLLM model receives the token ids of that very string (`TokensPrompt`), so
# the count is the prompt length of the request; the model compares the two before and after generation (vllm_model.py `_check_exact_counts`) and fails loudly on any difference.
"""Exact prompt size: `get()` -> `Exact` (render + count) when the run exports its tokenizer, else None (unit tests, `lighteval tasks inspect`: nothing is marked)."""

import hashlib
import json
import logging
import os
from typing import Callable, Iterable

from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)

ENV_TOKENIZER = "TITAN_EVAL_TOKENIZER"        # the model folder (titan_eval chat.py context_env)
ENV_RENDER = "TITAN_EVAL_RENDER"              # JSON of how the vLLM model renders and tokenizes (titan_eval chat.py render_env): the same values as the model config of the run
KEY = "exact_prompt_tokens"                   # doc.specific: the exact count the vLLM model has to see (vllm_model.py `_check_exact_counts`)
COUNT_CHUNK_CHARS = 16_000_000                # prompts counted in one batch call: up to this many characters (~5M tokens), so the list of ids stays small

# what a chat run does when the run exports no render settings: the tokenizer's own chat template, nothing extra, no special tokens added (VLLMModelConfig of a chat run: add_special_tokens False)
RENDER_DEFAULTS = {"chat_template": None, "chat_template_kwargs": {}, "system_prompt": None, "add_special_tokens": False, "trust_remote_code": False}


def render_settings() -> dict | None:
    """The run's rendering settings (None without TITAN_EVAL_TOKENIZER)."""
    path = os.environ.get(ENV_TOKENIZER)
    if not path:
        return None
    s = dict(RENDER_DEFAULTS)
    raw = os.environ.get(ENV_RENDER)
    if raw:
        s.update({k: v for k, v in json.loads(raw).items() if k in RENDER_DEFAULTS})
    s["chat_template_kwargs"] = dict(s["chat_template_kwargs"] or {})
    s["add_special_tokens"] = bool(s["add_special_tokens"])
    s["tokenizer"] = path
    return s


def load_tokenizer(settings: dict):
    """The tokenizer the vLLM model uses (VLLMModel._create_auto_tokenizer: vllm's get_tokenizer = AutoTokenizer.from_pretrained, `trust_remote_code` of the config), with the named chat template file
    replacing the tokenizer's own when the run has one."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(settings["tokenizer"], trust_remote_code=bool(settings["trust_remote_code"]))
    if settings["chat_template"]:
        with open(settings["chat_template"], encoding="utf-8") as f:
            tok.chat_template = f.read()
    return tok


class Exact:
    """Render a doc as the vLLM model does and count its tokens; counts are cached per rendered string (sha1)."""

    def __init__(self, settings: dict, tok):
        self.settings = settings
        self.tok = tok
        self.add_special_tokens = settings["add_special_tokens"]
        self.pm = PromptManager(True, tok, settings["system_prompt"], chat_template_kwargs=settings["chat_template_kwargs"])
        self._counts: dict[str, int] = {}
        self._fast_ok: bool | None = None          # None = not probed yet

    def render(self, doc: Doc) -> str:
        """The string the vLLM model tokenizes for this doc (`prepare_prompt`, the same call as vllm_model.py `_greedy_until`)."""
        return self.pm.prepare_prompt(doc)

    def _reference(self, texts: list[str]) -> list[int]:
        """The tokenizer call the vLLM model makes (`_greedy_until`: `self.tokenizer(context, add_special_tokens=...)`), batched."""
        return [len(one) for one in self.tok(texts, add_special_tokens=self.add_special_tokens)["input_ids"]]

    def _tokens(self, texts: list[str]) -> list[int]:
        """Token counts of a batch. A fast (Rust) tokenizer is counted through its backend's `encode_batch_fast`: the same ids as the call above (it only leaves out the character offsets, which
        the python wrapper converts for 100k-token prompts), about four times faster. The first batch of a run is counted both ways on its two shortest strings; a difference switches the run to the
        reference call (and says so), so the count can never silently differ from the vLLM model's."""
        backend = getattr(self.tok, "backend_tokenizer", None)
        fast = getattr(backend, "encode_batch_fast", None)
        if fast is None or self._fast_ok is False:
            return self._reference(texts)
        counts = [len(e) for e in fast(texts, add_special_tokens=self.add_special_tokens)]
        if self._fast_ok is None:
            probe = sorted(range(len(texts)), key=lambda i: len(texts[i]))[:2]
            ref = self._reference([texts[i] for i in probe])
            self._fast_ok = ref == [counts[i] for i in probe]
            if not self._fast_ok:
                logger.warning("exact_prompt: encode_batch_fast counts differ from the tokenizer call (%s vs %s): counting with the tokenizer call", [counts[i] for i in probe], ref)
                return self._reference(texts)
        return counts

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()

    def _fill(self, texts: Iterable[str]) -> None:
        """Count every string not yet in the cache, in batches of up to COUNT_CHUNK_CHARS characters (the strings are consumed one by one: a split of 100k-token prompts never sits in memory at once)."""
        chunk: dict[str, str] = {}
        size = 0

        def flush() -> None:
            nonlocal size
            if chunk:
                self._counts.update(zip(chunk, self._tokens(list(chunk.values()))))
                chunk.clear()
                size = 0

        for text in texts:
            k = self._key(text)
            if k in self._counts or k in chunk:
                continue
            chunk[k] = text
            size += len(text)
            if size >= COUNT_CHUNK_CHARS:
                flush()
        flush()

    def count_texts(self, texts: list[str]) -> list[int]:
        """Tokens of every string with the model's tokenizer call (`tok(text, add_special_tokens=...)["input_ids"]`, as `_greedy_until`), in batches."""
        self._fill(texts)
        return [self._counts[self._key(t)] for t in texts]

    def count_text(self, text: str) -> int:
        return self.count_texts([text])[0]

    def count(self, doc: Doc) -> int:
        """The prompt tokens of the request this doc becomes."""
        return self.count_text(self.render(doc))

    def prefetch(self, docs: Iterable[Doc]) -> None:
        """Render and count the docs into the cache (streaming)."""
        self._fill(self.render(d) for d in docs)


_INSTANCES: dict[tuple, tuple] = {}


def get() -> Exact | None:
    """The run's `Exact` (one per tokenizer + settings, tokenizer loaded once), None when the run did not export its tokenizer."""
    s = render_settings()
    if s is None:
        return None
    key = (s["tokenizer"], os.environ.get(ENV_RENDER) or "")
    hit = _INSTANCES.get(key)
    if hit is None or hit[0] is not load_tokenizer:           # (a unit test that patches `load_tokenizer` gets its own instance)
        hit = _INSTANCES[key] = (load_tokenizer, Exact(s, load_tokenizer(s)))
    return hit[1]


def reset() -> None:
    _INSTANCES.clear()


def stamp(spec: dict, n_prompt: int) -> dict:
    """Write the exact count into a doc's `specific`: `n_tokens` (what the details files and the docs have always called the prompt size) and `exact_prompt_tokens` (what the vLLM model checks)."""
    spec["n_tokens"] = n_prompt
    spec[KEY] = n_prompt
    return spec


def prefetcher(docs_of: Callable[[dict, str], Iterable[Doc]], active: Callable[[str], bool] = lambda name: True) -> Callable:
    """The `prefetch` hook of a prompt function (called by LightevalTask._get_docs_from_split with the rows of a split before the per-row calls): renders every row's doc with `docs_of(row, task name)` and
    counts them all in batches (the tokenizer's batch call runs on all cores), so that the per-row call finds its count in the cache. Does nothing without the run's tokenizer or when `active(task name)` is false."""

    def prefetch(rows, task_name: str | None = None) -> None:
        ex = get()
        if ex is None or not active(task_name or ""):
            return
        ex.prefetch(d for row in rows for d in docs_of(row, task_name or ""))

    return prefetch
