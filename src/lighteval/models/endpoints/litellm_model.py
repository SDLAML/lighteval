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
import os
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from json import JSONDecodeError

import requests
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached
from lighteval.utils.imports import is_package_available, requires


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if is_package_available("litellm"):
    import litellm
    from litellm import encode, supports_reasoning
    from litellm.caching.caching import Cache, LiteLLMCacheType
    from litellm.utils import ModelResponse as LitellmModelResponse
    from litellm.utils import get_max_tokens

    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)
    logging.getLogger("LiteLLM").handlers.clear()

    # Silence provider hint prints emitted directly via print() in LiteLLM internals.
    litellm.suppress_debug_info = True
    # Prefer an explicit cache root when provided; otherwise use a per-process
    # temp dir to avoid SQLite lock contention on shared filesystems when many
    # SLURM jobs run concurrently.
    import tempfile as _tempfile

    if os.environ.get("LITELLM_DISABLE_CACHE", "0").strip().lower() in ("1", "true", "yes"):
        litellm.disable_cache()
    else:
        _cache_dir = os.environ.get("LITELLM_CACHE_DIR")
        if _cache_dir:
            _cache_dir = os.path.expandvars(os.path.expanduser(_cache_dir))
            os.makedirs(_cache_dir, exist_ok=True)
        else:
            _cache_dir = _tempfile.mkdtemp(prefix="litellm_cache_")
        litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=_cache_dir)
else:
    from unittest.mock import Mock

    litellm = Mock()
    encode = Mock()
    LitellmModelResponse = Mock()


class LiteLLMModelConfig(ModelConfig):
    """Configuration class for LiteLLM unified API client.

    This configuration is used to connect to various LLM providers through the LiteLLM
    unified API. LiteLLM provides a consistent interface to multiple providers including
    OpenAI, Anthropic, Google, and many others.

    litellm doc: https://docs.litellm.ai/docs/

    Attributes:
        model_name (str):
            Model identifier. Can include provider prefix (e.g., "gpt-4", "claude-3-sonnet")
            or use provider/model format (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet").
        provider (str | None):
            Optional provider name override. If None, inferred from model_name.
            Examples: "openai", "anthropic", "google", "cohere", etc.
        base_url (str | None):
            Custom base URL for the API. If None, uses provider's default URL.
            Useful for using custom endpoints or local deployments.
        api_key (str | None):
            API key for authentication. If None, reads from environment variables.
            Environment variable names are provider-specific (e.g., OPENAI_API_KEY).
        concurrent_requests (int):
            Maximum number of concurrent API requests to execute in parallel.
            Higher values improve throughput by keeping the server busy; lower values
            are safer for rate-limited public APIs. Default is 64, which is appropriate
            for private vLLM deployments. For public APIs with strict rate limits (e.g.
            OpenAI free-tier), set this to 5–10. For large private servers (e.g. vLLM
            with DP=4 on H100s), 256–512 is reasonable.
        verbose (bool):
            Whether to enable verbose logging. Default is False.
        max_model_length (int | None):
            Maximum context length for the model. If None, infers the model's default max length.
        api_max_retry (int):
            Maximum number of retries for API requests. Default is 8.
        api_retry_sleep (float):
            Initial sleep time (in seconds) between retries. Default is 1.0.
        api_retry_multiplier (float):
            Multiplier for increasing sleep time between retries. Default is 2.0.
        timeout (float):
            Request timeout in seconds. Default is None (no timeout).
        generation_parameters (GenerationParameters, optional, defaults to empty GenerationParameters):
            Configuration parameters that control text generation behavior, including
            temperature, top_p, max_new_tokens, etc.
        system_prompt (str | None, optional, defaults to None): Optional system prompt to be used with chat models.
            This prompt sets the behavior and context for the model during evaluation.
        cache_dir (str, optional, defaults to "~/.cache/huggingface/lighteval"): Directory to cache the model.

    Example:
        ```python
        config = LiteLLMModelConfig(
            model_name="gpt-4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            concurrent_requests=5,
            generation_parameters=GenerationParameters(
                temperature=0.7,
                max_new_tokens=100
            )
        )
        ```
    """

    model_name: str
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    concurrent_requests: int = 256
    verbose: bool = False
    max_model_length: int | None = None
    extra_body: dict | None = None
    merge_reasoning_content_in_choices: bool = False
    use_chat_template: bool = True
    trust_remote_code: bool = False
    tokenizer_path: str | None = None
    """HuggingFace tokenizer name or local path for accurate token counting in
    loglikelihood tasks.  If None, falls back to the TOKENIZER_PATH env var, then
    to litellm's encode() which may use the wrong tokenizer for custom vLLM
    deployments and produce a shifted logprob window.
    Example: "/path/to/model" or "meta-llama/Llama-3.1-8B-Instruct"."""

    logprob_batch_size: int = 256
    """Number of (context, choice) pairs to score in a single batched text_completion
    call.  vLLM's /v1/completions accepts a list of prompt strings and returns one
    choice per prompt, reducing N×K HTTP round-trips to ceil(N×K / logprob_batch_size).
    Set to 1 to use per-pair requests (compatibility mode for servers that do not
    support batched prompts).  Tune together with concurrent_requests to match
    your vLLM server capacity."""

    api_max_retry: int = 8
    api_retry_sleep: float = 1.0
    api_retry_multiplier: float = 2.0
    timeout: float | None = None


@requires("litellm")
class LiteLLMClient(LightevalModel):
    _DEFAULT_MAX_LENGTH: int = 4096

    def __init__(self, config: LiteLLMModelConfig) -> None:
        """IMPORTANT: Your API keys should be set in the environment variables.
        If a base_url is not set, it will default to the public API.
        """
        self.config = config
        self.model = config.model_name
        self.provider = config.provider or config.model_name.split("/")[0]
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters
        self.concurrent_requests = config.concurrent_requests
        self._max_length = config.max_model_length
        self._enable_litellm_caching = _env_flag("LIGHTEVAL_LITELLM_CACHING", False)
        self.extra_body = config.extra_body
        self.merge_reasoning_content_in_choices = (
            config.merge_reasoning_content_in_choices
        )
        self.use_chat_template = config.use_chat_template
        self.logprob_batch_size = config.logprob_batch_size

        self.API_MAX_RETRY = config.api_max_retry
        self.API_RETRY_SLEEP = config.api_retry_sleep
        self.API_RETRY_MULTIPLIER = config.api_retry_multiplier
        self.timeout = config.timeout

        self._tokenizer = encode
        self.pairwise_tokenization = False
        litellm.drop_params = True
        litellm.verbose = config.verbose
        self.prompt_manager = PromptManager(
            use_chat_template=self.use_chat_template,
            tokenizer=self.tokenizer,
            system_prompt=config.system_prompt,
        )
        self._warned_empty_reasoning_only_response = threading.Event()
        self._warned_reasoning_tokens = threading.Event()
        self._warned_o1 = threading.Event()

        # Load HF tokenizer for accurate ctx_len in loglikelihood tasks.
        # litellm's encode() maps model names to tokenizers; for custom vLLM deployments
        # it often falls back to tiktoken or char/4, giving wrong token counts and
        # a shifted logprob window.  tokenizer_path (or RULER_TOKENIZER env var) fixes this.
        self._hf_tokenizer = None
        tokenizer_path = config.tokenizer_path or os.environ.get("TOKENIZER_PATH")
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer

                self._hf_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=config.trust_remote_code,
                )
                logger.info(f"Loaded HF tokenizer from: {tokenizer_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load HF tokenizer '{tokenizer_path}': {e}. Falling back to litellm encode()."
                )

        # Cache model-type flags once so per-request code avoids repeated lookups.
        try:
            self._is_reasoning_model: bool = supports_reasoning(config.model_name)
        except Exception:
            self._is_reasoning_model = False

        self._is_o1_model: bool = "o1" in config.model_name

        # Protects lazy initialisation of _max_length against concurrent writes.
        self._max_length_lock = threading.Lock()

        # Persistent thread pool — reused across all parallel API calls.
        self._executor = ThreadPoolExecutor(max_workers=self.concurrent_requests)

        # Initialize cache for tokenization and predictions
        self._cache = SampleCache(config)

    def cleanup(self):
        self._executor.shutdown(wait=False)

    def _prepare_stop_sequence(self, stop_sequence):
        """Prepare and validate stop sequence."""
        if self.provider == "anthropic":
            # Filter out whitespace-only stop sequences
            if stop_sequence:
                stop_sequence = [s for s in stop_sequence if s and s.strip()]
        return stop_sequence

    def _prepare_max_new_tokens(self, max_new_tokens) -> int | None:
        """Calculate completion tokens based on max_new_tokens."""
        if not max_new_tokens or max_new_tokens <= 0:
            return None

        if self._is_reasoning_model:
            # We need to allow more tokens to include reasoning tokens
            max_new_tokens = min(max_new_tokens * 10, self.max_length)

            if not self._warned_reasoning_tokens.is_set():
                logger.warning(
                    f"Reasoning model detected, increasing max_new_tokens to {max_new_tokens} to allow for reasoning tokens",
                )
                self._warned_reasoning_tokens.set()

        return max_new_tokens

    @staticmethod
    def _get_choice_text(choice) -> str | None:
        message = getattr(choice, "message", None)
        if message is not None:
            return getattr(message, "content", None)
        if hasattr(choice, "text"):
            return choice.text
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                return message.get("content")
            return choice.get("text")
        return None

    @staticmethod
    def _get_choice_reasoning(choice) -> str | None:
        message = getattr(choice, "message", None)
        if message is not None:
            return getattr(message, "reasoning_content", None)
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                return message.get("reasoning_content")
        return None

    def __call_api(
        self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence
    ):  # noqa: C901
        """Make API call with retries."""
        response = LitellmModelResponse()
        stop_sequence = self._prepare_stop_sequence(stop_sequence)
        max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)

        if return_logits and not self.provider == "openai":
            logger.warning(
                "Returning logits is not supported for this provider, ignoring."
            )

        # Prepare kwargs for completion call
        kwargs = {
            "model": self.model,
            # Explicit provider routing avoids LiteLLM provider inference errors
            # when model names are plain IDs or filesystem-like names.
            "custom_llm_provider": self.provider,
            "max_tokens": max_new_tokens,
            "logprobs": return_logits if self.provider == "openai" else None,
            "stop": stop_sequence,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "n": num_samples,
            "caching": self._enable_litellm_caching,
            "timeout": self.timeout,
            "extra_body": self.extra_body,
        }
        completion_call = litellm.completion
        if self.use_chat_template:
            kwargs["messages"] = prompt
            kwargs["response_format"] = {"type": "text"}
            kwargs["merge_reasoning_content_in_choices"] = (
                self.merge_reasoning_content_in_choices
            )
        else:
            kwargs["prompt"] = prompt
            completion_call = litellm.text_completion

        if self._is_o1_model:
            if not self._warned_o1.is_set():
                logger.warning(
                    "O1 models do not support temperature, top_p, stop sequence. Disabling."
                )
                self._warned_o1.set()
        else:
            kwargs.update(self.generation_parameters.to_litellm_dict())

        if self.use_chat_template and kwargs.get("max_completion_tokens", None) is None:
            kwargs["max_completion_tokens"] = max_new_tokens

        for attempt in range(self.API_MAX_RETRY):
            try:
                response = completion_call(**kwargs)
                content = self._get_choice_text(response.choices[0])
                reasoning_content = self._get_choice_reasoning(response.choices[0])

                if (
                    not content
                    and reasoning_content
                    and not self._warned_empty_reasoning_only_response.is_set()
                ):
                    logger.warning(
                        "Endpoint response contained reasoning_content but no final content. "
                        "This usually means the model is still in thinking mode; disable thinking for evals "
                        "(for example via extra_body.chat_template_kwargs.enable_thinking=false for Qwen3/vLLM)."
                    )
                    self._warned_empty_reasoning_only_response.set()

                # If response is empty, retry without caching (maybe the error is recoverable and solved with a retry)
                if not content:
                    logger.info("Response is empty, retrying without caching")
                    kwargs["caching"] = False
                    response = completion_call(**kwargs)
                    content = self._get_choice_text(response.choices[0])

                return response
            except litellm.BadRequestError as e:
                if "message" in e.__dict__:
                    error_string = "The response was filtered due to the prompt triggering Microsoft's content management policy"
                    if error_string in e.__dict__["message"]:
                        logger.warning(f"{error_string}. Returning empty response.")
                        return LitellmModelResponse()
            except Exception as e:
                wait_time = min(
                    64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                )  # Exponential backoff with max 64s
                # Keep retry logs single-line to avoid noisy provider help text spam.
                err_text = str(e).strip()
                err_text = (
                    err_text.splitlines()[0] if err_text else e.__class__.__name__
                )
                logger.warning(
                    f"Error in API call ({e.__class__.__name__}: {err_text}), "
                    f"waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait_time)

        logger.error(
            f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response."
        )
        return LitellmModelResponse()

    def __call_api_parallel(
        self,
        prompts,
        return_logits: bool | list[bool],
        max_new_tokens: int | list[int] | None,
        num_samples: int | list[int],
        stop_sequence: list[str] | None = None,
    ):
        return_logitss = (
            [return_logits for _ in prompts]
            if not isinstance(return_logits, list)
            else return_logits
        )
        max_new_tokenss = (
            [max_new_tokens for _ in prompts]
            if not isinstance(max_new_tokens, list)
            else max_new_tokens
        )
        num_sampless = (
            [num_samples for _ in prompts]
            if not isinstance(num_samples, list)
            else num_samples
        )
        stop_sequencess = [stop_sequence for _ in prompts]
        assert (
            len(prompts)
            == len(return_logitss)
            == len(max_new_tokenss)
            == len(num_sampless)
            == len(stop_sequencess)
        ), f"Length of prompts, return_logitss, max_new_tokenss, num_sampless, stop_sequences, system_prompts should be the same but are {len(prompts)}, {len(return_logitss)}, {len(max_new_tokenss)}, {len(num_sampless)}, {len(stop_sequencess)}"

        futures = {
            self._executor.submit(self.__call_api, p, rl, mn, ns, ss): i
            for i, (p, rl, mn, ns, ss) in enumerate(
                zip(
                    prompts,
                    return_logitss,
                    max_new_tokenss,
                    num_sampless,
                    stop_sequencess,
                )
            )
        }
        results = [None] * len(prompts)
        for future in tqdm(as_completed(futures), total=len(prompts)):
            results[futures[future]] = future.result()

        if None in results:
            raise ValueError(
                "Some entries are not annotated due to errors in annotate_p, please inspect and retry."
            )

        return results

    def estimate_context_length(self) -> int:
        def fallback():
            logger.warning(
                "Failed to fetch model endpoint info from OpenRouter, returning default max length."
            )
            return self._DEFAULT_MAX_LENGTH

        # If the model is used through openrouter, the actual model name comes after the prefix
        model_name = self.model.removeprefix("openrouter/")
        endpoint_info_response = requests.get(
            f"https://openrouter.ai/api/v1/models/{model_name}/endpoints",
            headers={},
        )
        if endpoint_info_response.ok:
            try:
                endpoint_info = endpoint_info_response.json()
                context_lengths = {
                    endpoint["provider_name"]: endpoint["context_length"]
                    for endpoint in endpoint_info["data"]["endpoints"]
                }

                if self.provider in context_lengths:
                    return context_lengths[self.provider]

                min_length = min(context_lengths.values())
                logger.warning(
                    f"Estimating model context length as the minimum context length from available OpenRouter providers: {min_length}"
                )
                return min_length
            except (KeyError, TypeError, ValueError, JSONDecodeError):
                return fallback()

        return fallback()

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(
        self,
        docs: list[Doc],
    ) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met.

        Args:
            docs (list[Doc]): List of documents containing the context for generation.

        Returns:
            list[ModelResponse]: list of generated responses.
        """
        dataset = GenerativeTaskDataset(
            requests=docs, num_dataset_splits=self.DATASET_SPLITS
        )
        results = []

        for split in tqdm(
            dataset.splits_iterator(),
            total=dataset.num_dataset_splits,
            desc="Splits",
            position=0,
            disable=self.disable_tqdm,
        ):
            if self.use_chat_template:
                contexts = [
                    self.prompt_manager.prepare_prompt_api(doc) for doc in split
                ]
            else:
                contexts = [self.prompt_manager.prepare_prompt(doc) for doc in split]
            max_new_tokens = split[0].generation_size  # could be none
            return_logits = split[0].use_logits

            # Left-truncate (OLMES-style): drop early few-shot examples when
            # the prompt + generation would exceed the model's context window.
            if not self.use_chat_template and max_new_tokens:
                max_ctx = self.max_length - max_new_tokens
                truncated = []
                for ctx in contexts:
                    if self._count_tokens(ctx) > max_ctx:
                        logger.warning(
                            f"Prompt too long (> {max_ctx} tokens for max_new_tokens={max_new_tokens}); left-truncating."
                        )
                        ctx = self._left_truncate_tokens(ctx, max_ctx)
                    truncated.append(ctx)
                contexts = truncated
            num_samples = split[0].num_samples
            stop_sequence = split[0].stop_sequences

            if num_samples > 1 and self.generation_parameters.temperature == 0:
                raise ValueError(
                    "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                )

            responses = self.__call_api_parallel(
                contexts, return_logits, max_new_tokens, num_samples, stop_sequence
            )

            for response, context in zip(responses, contexts):
                result: list[str] = [
                    (self._get_choice_text(choice) or "") for choice in response.choices
                ]
                reasonings: list[str | None] = [
                    self._get_choice_reasoning(choice) for choice in response.choices
                ]

                cur_response = ModelResponse(
                    # In empty responses, the model should return an empty string instead of None
                    text=result if result and result[0] else [""],
                    reasonings=reasonings,
                    input=context,
                )
                results.append(cur_response)

        return dataset.get_original_order(results)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def add_special_tokens(self) -> bool:
        return False

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model."""
        if self._max_length is not None:
            return self._max_length

        with self._max_length_lock:
            # Re-check inside the lock: another thread may have set it while we waited.
            if self._max_length is not None:
                return self._max_length

            try:
                max_tokens = get_max_tokens(self.model)
            except Exception:
                logger.error(
                    f"Unable to get the maximum sequence length for model {self.model} from litellm. Fetching information from OpenRouter instead."
                )
                max_tokens = self.estimate_context_length()

            self._max_length = max_tokens

        return self._max_length

    def _count_tokens(self, text: str) -> int:
        """Return token count using the HF tokenizer if configured, else litellm encode."""
        if self._hf_tokenizer is not None:
            return len(self._hf_tokenizer.encode(text, add_special_tokens=False))
        try:
            return len(encode(self.model, text))
        except Exception:
            return max(1, len(text) // 4)

    def _left_truncate_tokens(self, text: str, max_tokens: int) -> str:
        """Left-truncate text to at most max_tokens tokens (keeping the rightmost tokens).

        Avoids tokenize+decode roundtrip when no truncation is needed.
        """
        if self._hf_tokenizer is not None:
            ids = self._hf_tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                return text
            return self._hf_tokenizer.decode(
                ids[-max_tokens:], skip_special_tokens=False
            )
        # Fallback: character-ratio approximation
        total = self._count_tokens(text)
        if total <= max_tokens:
            return text
        keep_chars = max(1, int(len(text) * max_tokens / total))
        return text[-keep_chars:]

    def _score_single(self, prompt: str, choice_len: int) -> tuple[float, bool, list[float]]:
        """Score one (context + choice) string via a single echo-based completions call.

        Args:
            prompt: Full text (context + choice) to score.
            choice_len: Number of choice tokens; sliced from the right of the echoed logprobs,
                before the one generated token.  This approach is server-BOS-agnostic.

        Returns:
            (logprob_sum, is_greedy)
        """
        for attempt in range(self.API_MAX_RETRY):
            try:
                response = litellm.text_completion(
                    model=self.model,
                    custom_llm_provider=self.provider,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    prompt=prompt,
                    max_tokens=1,
                    echo=True,
                    logprobs=1,
                    temperature=0,
                    caching=self._enable_litellm_caching,
                    timeout=self.timeout,
                )
                break
            except Exception as e:
                wait = min(
                    64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                )
                err = str(e).splitlines()[0] if str(e) else e.__class__.__name__
                logger.warning(
                    f"logprob call failed ({err}), retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait)
        else:
            logger.error("logprob call failed after all retries, returning -inf")
            return float("-inf"), False, []

        logprobs_obj = response.choices[0].logprobs
        token_logprobs = logprobs_obj.token_logprobs or []

        # Slice choice tokens from the right: the response is
        #   [BOS?] [context tokens...] [choice tokens...] [1 generated token]
        # Anchoring from the right is BOS-agnostic and robust to any server-side
        # special-token prepending.
        end = len(token_logprobs) - 1  # exclude the one generated token
        start = max(0, end - choice_len)

        cont_logprobs = [lp for lp in token_logprobs[start:end] if lp is not None]
        logprob_sum = sum(cont_logprobs) if cont_logprobs else float("-inf")

        is_greedy = False
        top_logprobs = logprobs_obj.top_logprobs or []
        tokens = logprobs_obj.tokens or []
        cont_top = top_logprobs[start:end]
        cont_toks = tokens[start:end]
        if cont_top and cont_toks and len(cont_top) == len(cont_toks):
            is_greedy = all(tok in top for tok, top in zip(cont_toks, cont_top) if top)

        return logprob_sum, is_greedy, cont_logprobs

    def _score_batch(
        self, batch: list[tuple[int, int, str, int]]
    ) -> list[tuple[int, int, tuple[float, bool]]]:
        """Score a batch of (context+choice) strings in a single batched echo call.

        vLLM's /v1/completions accepts ``prompt`` as a list of strings and
        returns one choice per input prompt, so a single HTTP round-trip covers
        all pairs in the batch.

        Args:
            batch: list of (doc_idx, choice_idx, full_text, choice_len)

        Returns:
            list of (doc_idx, choice_idx, (logprob_sum, is_greedy))
        """
        if len(batch) == 1:
            di, ci, full_text, choice_len = batch[0]
            return [(di, ci, self._score_single(full_text, choice_len))]

        prompts = [full_text for _, _, full_text, _ in batch]

        for attempt in range(self.API_MAX_RETRY):
            try:
                response = litellm.text_completion(
                    model=self.model,
                    custom_llm_provider=self.provider,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    prompt=prompts,
                    max_tokens=1,
                    echo=True,
                    logprobs=1,
                    temperature=0,
                    caching=self._enable_litellm_caching,
                    timeout=self.timeout,
                )
                break
            except Exception as e:
                wait = min(
                    64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                )
                err = str(e).splitlines()[0] if str(e) else e.__class__.__name__
                logger.warning(
                    f"batch logprob call failed ({err}), retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait)
        else:
            logger.error(
                "batch logprob call failed after all retries, returning -inf for all items"
            )
            return [(di, ci, (float("-inf"), False, [])) for di, ci, _, _ in batch]

        # choices[i].index == i when prompt is a list with n=1 (OpenAI completions API).
        choices_by_idx = {c.index: c for c in response.choices}

        results = []
        for seq_idx, (di, ci, _, choice_len) in enumerate(batch):
            choice = choices_by_idx.get(seq_idx)
            if choice is None:
                logger.warning(
                    f"Missing response for batch item {seq_idx}, returning -inf"
                )
                results.append((di, ci, (float("-inf"), False, [])))
                continue

            logprobs_obj = choice.logprobs
            token_logprobs = logprobs_obj.token_logprobs or []

            end = len(token_logprobs) - 1  # exclude the one generated token
            start = max(0, end - choice_len)

            cont_logprobs = [lp for lp in token_logprobs[start:end] if lp is not None]
            logprob_sum = sum(cont_logprobs) if cont_logprobs else float("-inf")

            top_logprobs = logprobs_obj.top_logprobs or []
            tokens = logprobs_obj.tokens or []
            cont_top = top_logprobs[start:end]
            cont_toks = tokens[start:end]
            is_greedy = False
            if cont_top and cont_toks and len(cont_top) == len(cont_toks):
                is_greedy = all(
                    tok in top for tok, top in zip(cont_toks, cont_top) if top
                )

            results.append((di, ci, (logprob_sum, is_greedy, cont_logprobs)))

        return results

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Compute log p(continuation | context) for each choice via echo-based completions.

        Uses /v1/completions with echo=True and logprobs=1. Requires
        use_chat_template=False and a server that supports the completions
        endpoint (e.g. vLLM).

        All (doc, choice) pairs are grouped into batches of logprob_batch_size and
        each batch is sent as a single API call (prompt=[list of strings]), reducing
        N×K round-trips to ceil(N×K / logprob_batch_size).  Each batch future runs
        in the ThreadPoolExecutor; tune concurrent_requests to control how many
        batches are in flight at once.
        """
        if self.use_chat_template:
            raise ValueError(
                "LiteLLMClient.loglikelihood requires use_chat_template=False. "
                "The chat completions API does not support echo-based logprob extraction."
            )

        dataset = LoglikelihoodDataset(
            requests=docs, num_dataset_splits=self.DATASET_SPLITS
        )

        doc_list: list[Doc] = []
        context_list: list[str] = []

        for split in dataset.splits_iterator():
            for doc in split:
                context = self.prompt_manager.prepare_prompt(doc)
                doc_list.append(doc)
                context_list.append(context)

        # scored[doc_idx][choice_idx] = (logprob_sum, is_greedy, per_token_logprobs)
        scored: list[list[tuple[float, bool, list[float]]]] = [
            [(0.0, False, [])] * len(doc.choices) for doc in doc_list
        ]

        # Pre-compute token lengths once per context (not 4× per context via choices)
        # to avoid redundant BPE tokenization in the thread pool.
        context_lens: list[int] = [self._count_tokens(ctx) for ctx in context_list]

        # Pre-compute choice lengths once per unique choice text.
        _unique_choices: dict[str, int] = {}
        for doc in doc_list:
            for choice in doc.choices:
                if choice not in _unique_choices:
                    _unique_choices[choice] = self._count_tokens(choice)

        # Flat work list: (doc_idx, choice_idx, context, choice, context_len, choice_len)
        work = [
            (di, ci, context_list[di], choice, context_lens[di], _unique_choices[choice])
            for di, doc in enumerate(doc_list)
            for ci, choice in enumerate(doc.choices)
        ]

        def _score_batch_work(batch_items: list[tuple]) -> list[tuple]:
            """Score a batch of (context+choice) pairs in a single batched echo call."""
            prepared = []
            for di, ci, context, choice, ctx_len, ch_len in batch_items:
                total_len = ctx_len + ch_len
                prompt = context + choice
                # Left-truncate (OLMES-style): drop early few-shot examples when
                # the combined prompt exceeds the model's context window.
                # Reserve 1 token for the required output token (server rejects if
                # input_tokens + 1 > max_length).
                max_input = self.max_length - 1
                if total_len > max_input:
                    logger.warning(
                        f"Prompt too long ({total_len} tokens > {max_input} max); left-truncating."
                    )
                    prompt = self._left_truncate_tokens(prompt, max_input)
                prepared.append((di, ci, prompt, ch_len))
            return self._score_batch(prepared)

        # Chunk work into batches — each batch becomes a single API call with a list
        # of prompts, reducing N×K HTTP round-trips to ceil(N×K / logprob_batch_size).
        bs = self.logprob_batch_size
        batches = [work[i : i + bs] for i in range(0, len(work), bs)]
        futures = [self._executor.submit(_score_batch_work, batch) for batch in batches]
        for future in tqdm(
            as_completed(futures),
            total=len(batches),
            desc="Loglikelihoods",
            disable=self.disable_tqdm,
        ):
            for di, ci, result in future.result():
                scored[di][ci] = result

        results: list[ModelResponse] = []
        for doc, context, pairs in zip(doc_list, context_list, scored):
            logprobs = [lp for lp, _, _ in pairs]
            argmax = [g for _, g, _ in pairs]
            per_token_logprobs = [ptl for _, _, ptl in pairs]
            output_tokens = [[0] * _unique_choices[choice] for choice in doc.choices]
            results.append(
                ModelResponse(
                    input=context, logprobs=logprobs, argmax_logits_eq_gold=argmax,
                    output_tokens=output_tokens, per_token_logprobs=per_token_logprobs,
                )
            )

        return dataset.get_original_order(results)

    def _rolling_logprob(self, context: str) -> ModelResponse:
        """Score a single document via echo-based rolling logprob (used in parallel)."""
        # Left-truncate to max context length (OLMES-style).
        truncated = self._left_truncate_tokens(context, self.max_length - 1)
        if truncated is not context:
            logger.warning(
                f"Rolling logprob prompt too long (> {self.max_length} tokens); left-truncating."
            )
            context = truncated
        for attempt in range(self.API_MAX_RETRY):
            try:
                response = litellm.text_completion(
                    model=self.model,
                    custom_llm_provider=self.provider,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    prompt=context,
                    max_tokens=1,
                    echo=True,
                    logprobs=0,
                    temperature=0,
                    caching=self._enable_litellm_caching,
                    timeout=self.timeout,
                )
                logprobs_obj = response.choices[0].logprobs
                token_logprobs = logprobs_obj.token_logprobs or []
                valid_lps = [lp for lp in token_logprobs[:-1] if lp is not None]
                return ModelResponse(input=context, logprobs=valid_lps)
            except Exception as e:
                wait = min(
                    64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt)
                )
                err = str(e).splitlines()[0] if str(e) else e.__class__.__name__
                logger.warning(
                    f"rolling logprob call failed ({err}), retry {attempt + 1}/{self.API_MAX_RETRY}"
                )
                time.sleep(wait)

        logger.error(
            "Rolling logprob call failed after all retries, returning empty response."
        )
        return ModelResponse(input=context, logprobs=[])

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """Compute rolling log-likelihood (perplexity) over the full context.

        Sends the full text via echo=True and sums all per-token logprobs.
        Used for tasks like Lambada that measure BPB/perplexity over a passage.
        Requires use_chat_template=False.
        """
        if self.use_chat_template:
            raise ValueError(
                "LiteLLMClient.loglikelihood_rolling requires use_chat_template=False."
            )

        contexts = [self.prompt_manager.prepare_prompt(doc) for doc in docs]

        futures = {
            self._executor.submit(self._rolling_logprob, ctx): i
            for i, ctx in enumerate(contexts)
        }
        results = [None] * len(contexts)
        for future in tqdm(
            as_completed(futures),
            total=len(contexts),
            desc="Rolling loglikelihoods",
            disable=self.disable_tqdm,
        ):
            results[futures[future]] = future.result()

        return results
