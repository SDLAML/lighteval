# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

"""Offline API regression tests using the installed vLLM, without loading model weights."""

# ruff: noqa: E402 -- Check the optional backend before importing its APIs.

import asyncio
import inspect
import os
import queue
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from packaging.requirements import Requirement

from lighteval.utils.imports import is_package_available


if not is_package_available(Requirement("vllm>=0.19")):
    raise unittest.SkipTest("These API regression tests require vLLM >= 0.19")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, PreTrainedTokenizerFast
from vllm import LLM, SamplingParams
from vllm.logprobs import Logprob
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from lighteval.metrics.utils.llm_as_judge import JudgeLM
from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm import vllm_model as backend
from lighteval.models.vllm.vllm_model import AsyncVLLMModel, VLLMModel, VLLMModelConfig
from lighteval.tasks.requests import Doc


class TestVLLMCompatibility(unittest.TestCase):
    def make_model(self, cls=VLLMModel, **generation):
        model = cls.__new__(cls)
        model.config = VLLMModelConfig(
            model_name="local-test-model", generation_parameters=GenerationParameters(**generation)
        )
        model._max_length = None
        model.data_parallel_size = 1
        model.tensor_parallel_size = 1
        model.distributed_backend = "mp"
        model.use_chat_template = False
        model.prompt_manager = Mock()
        model.prompt_manager.prepare_prompt.side_effect = lambda doc: doc.query

        def generate(prompts, sampling_params, **kwargs):
            # Bind to the actual installed API, so obsolete arguments fail the test.
            inspect.signature(LLM.generate).bind(None, prompts, sampling_params=sampling_params, **kwargs)
            return prompts

        model.model = Mock()
        model.model.generate.side_effect = generate
        return model

    def test_import_without_optional_vllm(self):
        script = """
import sys
from unittest.mock import patch
from lighteval.utils import imports
available = imports.is_package_available
with patch.object(imports, "is_package_available", side_effect=lambda p: False if p == "vllm" else available(p)):
    sys.modules["vllm"] = None
    from lighteval.models.vllm.vllm_model import VLLMModelConfig
    assert VLLMModelConfig(model_name="test").model_name == "test"
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_tokenizer_and_model_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({"<eos>": 0, "hello": 1}, unk_token="<eos>")),
                eos_token="<eos>",
                unk_token="<eos>",
            )
            tokenizer.save_pretrained(directory)
            GPT2Config(
                vocab_size=2, n_embd=16, n_layer=1, n_head=1, n_positions=128, bos_token_id=0, eos_token_id=0
            ).save_pretrained(directory)
            config = VLLMModelConfig(model_name=directory, dtype="float16")
            model = self.make_model()
            loaded = model._create_auto_tokenizer(config)
            self.assertEqual(loaded.encode("hello", add_special_tokens=False), [1])
            self.assertEqual(loaded.pad_token, loaded.eos_token)
            engine = SimpleNamespace(llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)))
            with patch.object(backend, "LLM", return_value=engine):
                model._create_auto_model(config)
            model_config = AsyncEngineArgs(**model.model_args).create_model_config()
            self.assertEqual(model_config.max_model_len, 128)

    def test_sync_engine_arguments_and_inferred_length(self):
        model = self.make_model()
        engine = SimpleNamespace(llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)))

        def create(**kwargs):
            # LLM accepts **kwargs, but EngineArgs must recognize every supplied option.
            AsyncEngineArgs(**kwargs)
            return engine

        with patch.object(backend, "LLM", side_effect=create):
            self.assertIs(model._create_auto_model(model.config), engine)
        self.assertEqual(model.max_length, 4096)
        self.assertNotIn("swap_space", model.model_args)

    def test_async_engine_arguments_and_inferred_length(self):
        model = self.make_model(AsyncVLLMModel)
        engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096))
        with patch.object(backend.AsyncLLM, "from_engine_args", return_value=engine) as create:
            self.assertIs(model._create_auto_model(model.config), engine)
        self.assertIsInstance(create.call_args.args[0], AsyncEngineArgs)
        self.assertEqual(model.max_length, 4096)

    def test_explicit_length_is_preserved(self):
        model = self.make_model()
        model._max_length = 1024
        with patch.object(backend, "LLM", return_value=object()):
            model._create_auto_model(model.config)
        self.assertEqual(model.max_length, 1024)

    def test_token_prompt_generation_and_stop_buffer(self):
        model = self.make_model(temperature=0.7)
        outputs = model._generate([[1, 2], [3]], max_new_tokens=8, stop_tokens=["STOP"], num_samples=2)
        self.assertEqual(outputs, [{"prompt_token_ids": [1, 2]}, {"prompt_token_ids": [3]}])
        params = model.model.generate.call_args.kwargs["sampling_params"]
        self.assertIsInstance(params, SamplingParams)
        self.assertEqual(params.output_text_buffer_length, 3)
        self.assertEqual(params.n, 2)
        self.assertEqual(params.max_tokens, 8)

    def test_none_stop_is_normalized(self):
        model = self.make_model()
        model._generate([[1]], max_new_tokens=8)
        self.assertEqual(model.model.generate.call_args.kwargs["sampling_params"].stop, [])

    def test_prompt_logprobs_bypass_prefix_cache(self):
        model = self.make_model(min_new_tokens=10, stop_tokens=["STOP"])
        model._generate([[1, 2]], generate=False)
        params = model.model.generate.call_args.kwargs["sampling_params"]
        self.assertTrue(params.skip_reading_prefix_cache)
        self.assertEqual(params.prompt_logprobs, 1)
        self.assertEqual(params.max_tokens, 1)
        self.assertEqual(params.min_tokens, 0)
        self.assertFalse(params.detokenize)
        self.assertEqual(params.stop, [])

    def test_multiple_choice_scoring_with_vllm_logprobs(self):
        model = self.make_model()
        model.pairwise_tokenization = False
        model.tok_encode_pair = Mock(return_value=([[1], [1]], [[2, 3], [4]]))
        outputs = [
            SimpleNamespace(prompt_logprobs=[None, {2: Logprob(-0.2, rank=1)}, {3: Logprob(-0.3, rank=1)}]),
            SimpleNamespace(prompt_logprobs=[None, {4: Logprob(-1.0, rank=2)}]),
        ]
        model.model.generate.side_effect = lambda *args, **kwargs: outputs
        doc = Doc(query="Question", choices=["AB", "C"], gold_index=0)
        responses = model._loglikelihood_tokens([doc])
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].logprobs, [-0.5, -1.0])
        self.assertEqual(responses[0].per_token_logprobs, [[-0.2, -0.3], [-1.0]])
        self.assertEqual(responses[0].argmax_logits_eq_gold, [True, False])
        self.assertTrue(model.model.generate.call_args.kwargs["sampling_params"].skip_reading_prefix_cache)

    def test_greedy_multiple_samples_are_rejected(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "num_samples > 1"):
            model._generate([[1]], max_new_tokens=8, num_samples=2)

    def test_multiprocessing_preserves_order_and_scoring_parameters(self):
        model = self.make_model()
        model.data_parallel_size = 2
        model.model_args = {"model": "local-test-model"}
        worker_calls = []

        def generate(prompts, sampling_params):
            worker_calls.append(sampling_params)
            return prompts

        def process(*, target, args):
            return SimpleNamespace(start=lambda: target(*args), join=lambda: None)

        context = SimpleNamespace(Queue=queue.Queue, Process=process)
        engine = SimpleNamespace(generate=generate)
        with (
            patch.object(backend.mp, "get_context", return_value=context),
            patch("vllm.LLM", return_value=engine),
            patch.dict(os.environ),
        ):
            outputs = model._generate([[1], [2], [3]], generate=False)
        self.assertEqual(outputs, [{"prompt_token_ids": [i]} for i in (1, 2, 3)])
        self.assertEqual(len(worker_calls), 2)
        self.assertTrue(all(p.skip_reading_prefix_cache for p in worker_calls))

    def test_async_generation_normalizes_sampling_parameters(self):
        model = self.make_model(AsyncVLLMModel)
        doc = Doc(query="hello", choices=[], gold_index=0, generation_size=8, stop_sequences=["STOP"])
        final = object()
        calls = []

        async def generate(**kwargs):
            inspect.signature(AsyncLLM.generate).bind(None, **kwargs)
            calls.append(kwargs)
            yield object()
            yield final

        model.model.generate = generate
        self.assertIs(asyncio.run(model._async_one_item(0, doc, generative=True)), final)
        params = calls[0]["sampling_params"]
        self.assertEqual(params.output_text_buffer_length, 3)
        self.assertEqual(params.max_tokens, 8)
        self.assertEqual(calls[0]["request_id"], "generative_0")

    def test_async_cleanup_shuts_down_engine(self):
        model = self.make_model(AsyncVLLMModel)
        engine = model.model
        with patch.object(backend, "destroy_distributed_environment"), patch.object(backend.torch.cuda, "empty_cache"):
            model.cleanup()
            model.cleanup()
        engine.shutdown.assert_called_once_with()
        self.assertIsNone(model.model)

    def test_judge_uses_token_prompts_and_engine_tokenizer(self):
        judge = JudgeLM.__new__(JudgeLM)
        judge.backend = "vllm"
        judge.pipe = None
        judge.model = "local-test-model"
        judge.max_tokens = 8
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [1, 2]
        pipe = Mock()
        pipe.get_tokenizer.return_value = tokenizer

        def generate(prompts, **kwargs):
            inspect.signature(LLM.generate).bind(None, prompts, **kwargs)
            self.assertEqual(prompts, [{"prompt_token_ids": [1, 2]}])
            return [SimpleNamespace(outputs=[SimpleNamespace(text="yes")])]

        pipe.generate.side_effect = generate
        with patch("vllm.LLM", return_value=pipe):
            call = judge._JudgeLM__lazy_load_client()
        self.assertIs(judge.tokenizer, tokenizer)
        self.assertEqual(call([[{"role": "user", "content": "hello"}]]), ["yes"])


if __name__ == "__main__":
    unittest.main()
