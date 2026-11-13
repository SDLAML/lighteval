"""
name:
AIME 2024 / 2025 / 2026 and HMMT February 2026, base-model protocol (n = 10 samples, pass@1 and pass@10)

dataset:
HuggingFaceH4/aime_2024, yentinglin/aime_2025, MathArena/aime_2026, MathArena/hmmt_feb_2026

abstract:
Competition math (30 / 30 / 30 AIME problems with integer answers 0-999, 33 HMMT problems with LaTeX answers), 0-shot, in the
format of the code evaluations: n = 10 samples per problem, sampled (the registry file sets temperature 0.6 and top_p 0.95; at
temperature 0 lighteval refuses n > 1), at most 4096 new tokens (an explicit parameter, set in the registry file; the task default
is the same). The score is the fork's math scorer of math_500 / math:gen (Metrics.expr_gold_metric: boxed answer first, then an
expression; sympy equivalence, precision 5), applied to each sample: pass@1 = the unbiased estimator with n = 10, k = 1 (= the mean
of the 10 samples) and pass@10 = any of the 10 samples correct (n = k = 10). The prompt is the instruct-style AIME / HMMT prompt of
Oleg Filatov's aime.py ("... final answer is: $\\boxed{ANSWER}$ ...") with the "Problem:" / "Solution:" cues of our math_500, so a
base model starts a solution instead of extending the problem; generation stops where a new problem would begin: a line that starts
with "Problem:" or with the repeated instruction (version 1; version 0 stopped at "Problem:" anywhere and cut the model's own heading
"**Understanding the Problem:**", 1.3-4% of the samples).
AIME 2024 gold answers carry leading zeros ("025", 7 of 30); the scorer reads "025" and "25" as the same number (tested).
aime.py / hmmt.py are unchanged (aime24, aime25: n = 1, no cap; aime26, hmmt_feb_2026: n = 10, 32768 tokens, no pass@10).

languages:
english

tags:
math, reasoning

paper:
https://maa.org/aime-thresholds-are-available/
"""

import copy
from functools import partial

import numpy as np

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation, SamplingMetric
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


N_SAMPLES = 10
KS = (1, 10)
MAX_NEW_TOKENS = 4096
# Version 1 (review fix 11-13): a stop string must only match where a NEW problem begins, i.e. at the start of a line. The prompt puts
# "\n\nProblem:\n" between the instruction and the problem, so a base model that goes on in the same format writes "\n\nProblem:\n<next
# problem>" or the whole instruction again after its answer. Version 0 used "Problem:" / "Solve the following math problem" anywhere in
# the text: the model's own heading "1. **Understanding the Problem:**" was cut after "the" (dead sample without an answer: 4 / 9 / 12 / 7
# of 300 / 300 / 300 / 330 samples of Qwen3-1.7B-Base for AIME 24 / 25 / 26 / HMMT). One "\n" (not two) also catches a continuation that
# follows its answer line without a blank line. vLLM compares the stop strings with the generated text only (never the prompt).
STOP_SEQUENCES = ["\nProblem:", "\nSolve the following math problem"]
TASK_VERSION = 1
PROMPT_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: "
    "'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final "
    "number or expression that solves the problem. Think step by step before answering.\n\nProblem:\n{problem}\n\nSolution:"
)


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. 2021: 1 - C(n - c, k) / C(n, k). With fewer than k samples there is no estimate: any correct sample counts."""
    if n < k:
        return float(c > 0)
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


class MathPassAt(SamplingMetric, SampleLevelComputation):
    """pass@k for several k from the same n samples, scored with the fork's expr_gold_metric extraction (one extraction pass)."""

    def __init__(self, n: int = N_SAMPLES, ks: tuple[int, ...] = KS):
        # the very scorer of math_500 / math:gen / gsm8k: boxed first, then expressions; sympy equivalence at precision 5
        super().__init__(
            strip_strings=True, sample_scoring_function=copy.deepcopy(Metrics.expr_gold_metric.value.sample_level_fn)
        )
        self.n = n
        self.ks = tuple(ks)

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict[str, float]:
        if len(doc.get_golds()) > 1:
            raise ValueError("pass@k needs exactly one gold")
        gold_doc = Doc(choices=[self.preprocess(g) for g in doc.choices], query=doc.query, gold_index=doc.gold_index)
        predictions = list(model_response.final_text)[: self.n]
        scores = [
            self.compute_score(doc=gold_doc, model_response=ModelResponse(text=[self.preprocess(prediction)]))
            for prediction in predictions
        ]
        correct = sum(1 for score in scores if score == 1)
        return {f"pass@{k}": unbiased_pass_at_k(len(scores), correct, k) for k in self.ks}

    def num_samples(self) -> int:
        return self.n


PASS_AT = SampleLevelMetricGrouping(
    metric_name=[f"pass@{k}" for k in KS],
    sample_level_fn=MathPassAt(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={f"pass@{k}": np.mean for k in KS},
    higher_is_better={f"pass@{k}": True for k in KS},
)


def aime_gold(answer) -> str:
    """AIME answers are the integers 0-999 ('HuggingFaceH4/aime_2024' stores them zero-padded, '025'; the scorer reads it as 25)."""
    return str(answer).strip()


def hmmt_gold(answer) -> str:
    """HMMT answers are bare LaTeX; wrapped in $...$ so the math extractor reads them (no HMMT answer contains '$')."""
    return f"${answer}$"


def math_base_prompt(line, task_name: str = None, *, gold_of, specific_of):
    return Doc(
        task_name=task_name,
        query=PROMPT_TEMPLATE.format(problem=line["problem"]),
        choices=[gold_of(line["answer"])],
        gold_index=0,
        specific=specific_of(line),
    )


def _config(name, repo, revision, gold_of, specific_of):
    return LightevalTaskConfig(
        name=name,
        prompt_function=partial(math_base_prompt, gold_of=gold_of, specific_of=specific_of),
        hf_repo=repo,
        hf_subset="default",
        hf_revision=revision,
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=MAX_NEW_TOKENS,
        metrics=[PASS_AT],
        stop_sequence=STOP_SEQUENCES,
        version=TASK_VERSION,  # 1: stop strings only at the start of a line (0: "Problem:" anywhere cut "**Understanding the Problem:**")
    )


def _aime24_specific(line):
    return {"id": line["id"], "url": line["url"]}


def _aime25_specific(line):
    return {"id": line["id"], "url": line["url"]}


def _matharena_specific(line):
    return {"problem_idx": line["problem_idx"]}


TASKS_TABLE = [
    _config("aime24_base", "HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a", aime_gold, _aime24_specific),
    _config("aime25_base", "yentinglin/aime_2025", "6f71d77b0b89b9dabe07ab466c51df33f514df7f", aime_gold, _aime25_specific),
    _config("aime26_base", "MathArena/aime_2026", "d61f486805306e6e6b5913504c07763a99e1b112", aime_gold, _matharena_specific),
    _config("hmmt_feb_2026_base", "MathArena/hmmt_feb_2026", "589a53fe152febc0f4f99f3e561007bf8da8e664", hmmt_gold, _matharena_specific),
]
