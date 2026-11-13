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

"""Persistent data-parallel vLLM workers (multiprocessing backend).

The old multiprocessing path started one process per GPU on EVERY `_generate` call, and each process built a new vLLM
engine, generated, and exited. A lighteval call with several generation splits (one per distinct generation size /
stop sequence) therefore paid the engine bring-up once per split and GPU (about 70 s each), and the requests were
pickled into the `Process` arguments, so `Process.start()` blocked until each child had finished its imports.

Here one engine per GPU lives for the whole lighteval call:

* `DPPool.start()` spawns all workers at once; their only argument is the (small) engine-args dict, then waits until
  every engine is built.
* `DPPool.generate()` sends every worker its share over a queue and collects the answers.
* `DPPool.shutdown()` (called from `VLLMModel.cleanup()`) stops the workers.

A worker exception is sent back with its traceback and raised in the parent (the pool is torn down, because the other
workers are still busy with the failed job). A dead worker (killed, out of memory, ...) is noticed by a liveness check
on the result queue, so the parent never waits forever. Workers are not daemonic (vLLM spawns its own engine process); they exit on shutdown() and when the parent disappears.

This module imports only the standard library, so the (spawned) workers and the unit tests do not pay for importing
vLLM unless an engine is really built.
"""

import gc
import importlib
import itertools
import logging
import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
from collections import namedtuple
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)

# What `_loglikelihood_tokens` needs from a vLLM `Logprob` (same attribute names).
LP = namedtuple("LP", ["logprob", "rank"])


@dataclass
class GenCompletion:
    """One generation of a request (the part of vLLM's CompletionOutput that lighteval uses)."""

    text: str
    token_ids: list
    finish_reason: str | None = None  # "stop" or "length"


@dataclass
class GenOutput:
    """A slim stand-in for vLLM's RequestOutput: no logprob dicts for the prompt, no metrics, no prompt text."""

    prompt_token_ids: list
    outputs: list = field(default_factory=list)
    prompt_logprobs: list | None = None


class DPWorkerError(RuntimeError):
    """A data-parallel worker failed (exception in the worker, or the worker died)."""


def slim_output(request_output, keep_last: int | None = None) -> tuple:
    """RequestOutput -> plain tuple `(completions, prompt_logprobs)`, cheap to pickle.

    completions: `[(text, token_ids, finish_reason), ...]`, one per generation (n).
    prompt_logprobs: None, or one dict per prompt position `{token_id: LP(logprob, rank)}` (first entry None); with
    `keep_last=k` only the last k positions are kept (the loglikelihood path reads only the continuation's positions,
    counted from the end).
    """
    completions = [
        (c.text, list(c.token_ids), getattr(c, "finish_reason", None)) for c in request_output.outputs
    ]
    prompt_logprobs = getattr(request_output, "prompt_logprobs", None)
    if prompt_logprobs is not None:
        if keep_last is not None:
            prompt_logprobs = prompt_logprobs[-keep_last:] if keep_last > 0 else []
        prompt_logprobs = [
            None if d is None else {tok: LP(lp.logprob, lp.rank) for tok, lp in d.items()} for d in prompt_logprobs
        ]
    return completions, prompt_logprobs


def rebuild_output(slim: tuple, prompt_token_ids: list) -> GenOutput:
    completions, prompt_logprobs = slim
    return GenOutput(
        prompt_token_ids=prompt_token_ids,
        outputs=[GenCompletion(text=t, token_ids=ids, finish_reason=fr) for t, ids, fr in completions],
        prompt_logprobs=prompt_logprobs,
    )


def interleave(per_worker: list[list]) -> list:
    """Inverse of `more_itertools.distribute(n, items)`: the items back in their original order."""
    return [x for x in itertools.chain.from_iterable(itertools.zip_longest(*per_worker)) if x is not None]


class VllmEngine:
    """The engine a worker owns: a vLLM `LLM` returning slim outputs. (Tests plug in a fake with the same methods.)"""

    def __init__(self, gpu_id: int, model_args: dict):
        from vllm import LLM

        self.gpu_id = gpu_id
        self.llm = LLM(**model_args)

    def generate(self, requests: list, sampling_kwargs: dict, keep_last: list | None = None) -> list:
        from vllm import SamplingParams

        sampling_params = SamplingParams(**sampling_kwargs)
        # a progress bar from worker 0 only (all workers get a similar share)
        outputs = self.llm.generate(requests, sampling_params=sampling_params, use_tqdm=self.gpu_id == 0)
        keep_last = keep_last if keep_last is not None else [None] * len(outputs)
        return [slim_output(o, k) for o, k in zip(outputs, keep_last)]

    def shutdown(self):
        try:
            core = getattr(self.llm.llm_engine, "engine_core", None)
            if core is not None:
                core.shutdown()
        except Exception:  # best effort: the process is about to exit
            logger.warning("vLLM engine shutdown failed:\n%s", traceback.format_exc())
        self.llm = None
        gc.collect()


def _make_engine(gpu_id: int, model_args: dict, engine_factory: str | None):
    if engine_factory is None:
        return VllmEngine(gpu_id, model_args)
    module_name, _, func_name = engine_factory.partition(":")  # "package.module:function"
    return getattr(importlib.import_module(module_name), func_name)(gpu_id, model_args)


def _dp_worker(gpu_id: int, devices: str, model_args: dict, engine_factory: str | None, req_q, res_q):
    """Worker process. Must be at module level (spawn pickles it by name).

    Messages out: `(gpu_id, job_id, kind, payload)` with kind "ready" | "ok" | "error" (payload: the traceback text).
    Messages in: `(job_id, requests, sampling_kwargs, keep_last)`, or None to stop.
    """
    engine = None
    parent = mp.parent_process()

    def _on_sigterm(signum, frame):
        raise SystemExit(0)  # unwinds to the finally below, which shuts the engine (and its core process) down

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            engine = _make_engine(gpu_id, model_args, engine_factory)
        except Exception:
            res_q.put((gpu_id, 0, "error", "engine start failed:\n" + traceback.format_exc()))
            return
        res_q.put((gpu_id, 0, "ready", None))
        while True:
            try:
                msg = req_q.get(timeout=5)
            except queue.Empty:
                if parent is not None and not parent.is_alive():
                    return  # the parent is gone (e.g. killed): do not keep the GPU
                continue
            if msg is None:
                return
            job_id, requests, sampling_kwargs, keep_last = msg
            try:
                outputs = engine.generate(requests, sampling_kwargs, keep_last) if requests else []
                res_q.put((gpu_id, job_id, "ok", outputs))
            except Exception:
                res_q.put((gpu_id, job_id, "error", traceback.format_exc()))
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                logger.warning("worker %s: engine shutdown failed:\n%s", gpu_id, traceback.format_exc())


class DPPool:
    """One persistent engine per GPU. Not thread safe (lighteval calls it from one thread)."""

    def __init__(
        self,
        model_args: dict,
        n_workers: int,
        engine_factory: str | None = None,
        poll_seconds: float = 5.0,
    ):
        self.model_args = model_args
        self.n_workers = n_workers
        self.engine_factory = engine_factory
        self.poll_seconds = poll_seconds
        self._procs: list = []
        self._req_qs: list = []
        self._res_q = None
        self._job_ids = itertools.count(1)

    @property
    def started(self) -> bool:
        return bool(self._procs)

    def _device_lists(self) -> list[str]:
        """CUDA_VISIBLE_DEVICES for each worker: consecutive slices of the devices this process may use
        (tensor_parallel_size * pipeline_parallel_size devices per worker)."""
        width = int(self.model_args.get("tensor_parallel_size", 1)) * int(
            self.model_args.get("pipeline_parallel_size", 1)
        )
        need = self.n_workers * width
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or not visible.strip():
            ids = [str(i) for i in range(need)]
        else:
            ids = [x.strip() for x in visible.split(",") if x.strip()]
            if len(ids) < need:
                raise DPWorkerError(
                    f"data_parallel_size={self.n_workers} x {width} device(s) per replica needs {need} GPUs, but "
                    f"CUDA_VISIBLE_DEVICES={visible!r} lists {len(ids)}"
                )
        return [",".join(ids[g * width : (g + 1) * width]) for g in range(self.n_workers)]

    def start(self) -> None:
        if self.started:
            return
        devices = self._device_lists()
        # spawn (not fork) because CUDA must not be initialised in a forked child
        ctx = mp.get_context("spawn")
        self._res_q = ctx.Queue()
        self._req_qs = [ctx.Queue() for _ in range(self.n_workers)]
        t0 = time.time()
        try:
            for g in range(self.n_workers):
                p = ctx.Process(
                    target=_dp_worker,
                    args=(g, devices[g], self.model_args, self.engine_factory, self._req_qs[g], self._res_q),
                    # NOT daemonic: vLLM's engine starts its own EngineCore process, and daemonic processes may not have children
                    # (AssertionError on the node, 11-13). Workers still exit on shutdown() and when the parent disappears (poll).
                    daemon=False,
                    name=f"lighteval-dp-{g}",
                )
                p.start()
                self._procs.append(p)
            logger.info("Started %d data-parallel vLLM workers, waiting for the engines", self.n_workers)
            self._gather(0, "ready")
        except BaseException:
            self.shutdown(force=True)
            raise
        logger.info("All %d data-parallel engines are ready (%.0f s)", self.n_workers, time.time() - t0)

    def generate(self, shares: list[list], sampling_kwargs: dict, keep_last: list[list] | None = None) -> list[list]:
        """Send `shares[g]` to worker g; return its slim outputs, one list per worker (in worker order)."""
        if not self.started:
            self.start()
        assert len(shares) == self.n_workers
        dead = [g for g, p in enumerate(self._procs) if not p.is_alive()]
        if dead:
            self.shutdown(force=True)
            raise DPWorkerError(f"data-parallel worker(s) {dead} are not running")
        job_id = next(self._job_ids)
        for g, q in enumerate(self._req_qs):
            q.put((job_id, shares[g], sampling_kwargs, keep_last[g] if keep_last is not None else None))
        got = self._gather(job_id, "ok")
        return [got[g] for g in range(self.n_workers)]

    def _next_message(self) -> tuple:
        """Next result message; raises DPWorkerError if a worker has died and nothing else is coming."""
        while True:
            try:
                return self._res_q.get(timeout=self.poll_seconds)
            except queue.Empty:
                pass
            dead = [(g, p.exitcode) for g, p in enumerate(self._procs) if not p.is_alive()]
            if not dead:
                continue
            try:  # an error message the worker sent just before exiting may still be on its way
                return self._res_q.get(timeout=2.0)
            except queue.Empty:
                pass
            self.shutdown(force=True)
            raise DPWorkerError(
                "data-parallel worker(s) died without a result (index, exit code): "
                + ", ".join(f"{g}: {code}" for g, code in dead)
                + " (a negative code is the signal that killed it; -9 is usually the out-of-memory killer)"
            )

    def _gather(self, job_id: int, kind: str) -> dict:
        got: dict = {}
        while len(got) < self.n_workers:
            gpu_id, jid, k, payload = self._next_message()
            if k == "error":
                self.shutdown(force=True)  # the other workers are still on this job: start clean next time
                raise DPWorkerError(f"data-parallel worker {gpu_id} failed:\n{payload}")
            if jid != job_id or k != kind:
                continue  # stale message of an earlier job
            got[gpu_id] = payload
        return got

    def shutdown(self, force: bool = False, timeout: float = 60.0) -> None:
        """Stop the workers: ask them to exit (wait up to `timeout`), then terminate, then kill. Safe to call twice."""
        procs, self._procs = self._procs, []
        req_qs, self._req_qs = self._req_qs, []
        res_q, self._res_q = self._res_q, None
        if not procs:
            return
        if not force:
            for q in req_qs:
                try:
                    q.put(None)
                except Exception:
                    pass
            deadline = time.time() + timeout
            for p in procs:
                p.join(max(0.0, deadline - time.time()))
        for p in procs:
            if p.is_alive():
                p.terminate()  # SIGTERM: the worker shuts its engine down
        for p in procs:
            p.join(30)
            if p.is_alive():
                logger.warning("data-parallel worker %s did not stop on SIGTERM, killing it", p.name)
                p.kill()
                p.join(10)
        for q in [*req_qs, res_q]:
            if q is not None:
                try:
                    q.cancel_join_thread()  # never block the parent's exit on a queue nobody reads
                    q.close()
                except Exception:
                    pass
