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
name:
Chat evals (titan_eval, post-trained models): IFEval, IFBench, MATH-500, AIME 2024 / 2025 / 2026, HMMT Feb 2026, GPQA-Diamond, MMLU-Pro, MMLU-Redux

dataset:
google/IFEval, allenai/IFBench_test, HuggingFaceH4/MATH-500, HuggingFaceH4/aime_2024, yentinglin/aime_2025, MathArena/aime_2026, MathArena/hmmt_feb_2026, Idavidrein/gpqa, TIGER-Lab/MMLU-Pro, edinburgh-dawg/mmlu-redux-2.0

abstract:
Instruction prompts, 0-shot, for models used through their chat template (the prompt text is the user message; the template, its system
prompt and the thinking switch come from the run's model settings). Tasks are named `<name>_chat`; the base twins keep their names.
The scored text is what follows the LAST </think> (run with --strict-reasoning: an unclosed block gives an empty answer). Samples per prompt:
AIME / HMMT 32, GPQA-Diamond 8, MATH-500 4, the rest 1 (the task's default; the run's `num_samples` replaces it for the tasks that sample
more than once). Token budget: `generation_size` is the default (32,768); runs set gen.max_new_tokens. Metrics: lighteval/metrics/chat_metrics.py
(math: strict `\\boxed` avg@n, lenient avg@n_lenient, pass@k; multiple choice: anchored `Answer: X` acc, any-letter acc_unanchored, pass@k;
IFEval / IFBench: the base checkers, strict and loose, prompt and instruction level).
Math prompt = the instruction of the fork's aime.py / hmmt.py (no "Problem:" / "Solution:" cues: a chat model starts its answer by itself).
Multiple choice prompt = the instruction of gpqa:diamond ("Answer: $LETTER", "Think step by step before answering") with the letters of THIS
question (mmlu_pro:cot said ABCD for 10 options). GPQA option order = the md5 shuffle of gpqa.py (the upstream instruct prompt shuffled with an
unseeded random.randint: another order in every run). The Doc has no `instruction` (the prompt manager would re-attach it without the blank line).
MMLU-Redux: the questions MMLU-Redux 2.0 marks "ok" (as mmlu_redux:mcf_em), subjects as subtasks mmlu_redux_chat:<subject>.

languages:
english

tags:
chat, reasoning, instruction-following, math, multiple-choice

paper:
https://arxiv.org/abs/2505.09388
"""

import hashlib
import logging
import random
from functools import partial
from string import ascii_uppercase

from lighteval.metrics.chat_metrics import chat_choice_metric, chat_math_metric
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


logger = logging.getLogger(__name__)

BUDGET = 32768          # default max new tokens (the run's gen.max_new_tokens replaces it)
N_AIME = 32             # owner 11-13: avg@32 for AIME / HMMT (the Qwen3 report uses 64, "too much")
N_GPQA = 8
N_MATH500 = 4

# ---------------------------------------------------------------------------------------------------------- math
MATH_PROMPT = (
    "Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: "
    "'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final "
    "number or expression that solves the problem. Think step by step before answering.\n\n{problem}"
)


def _aime_gold(answer) -> str:
    return str(answer).strip()        # '070' and '70' are the same number for the scorer (tested)


def _hmmt_gold(answer) -> str:
    return f"${answer}$"              # bare LaTeX in the dataset, wrapped for the extractor (no HMMT answer contains '$')


def math_chat_prompt(line, task_name: str = None, *, gold_of, specific_of=None):
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT.format(problem=line["problem"].strip()),
        choices=[gold_of(line)],
        gold_index=0,
        specific=specific_of(line) if specific_of else None,
    )


def _math_cfg(name, repo, revision, n, gold_of, specific_of=None, split="train", subset="default"):
    return LightevalTaskConfig(
        name=name,
        prompt_function=partial(math_chat_prompt, gold_of=gold_of, specific_of=specific_of),
        hf_repo=repo,
        hf_subset=subset,
        hf_revision=revision,
        hf_avail_splits=[split],
        evaluation_splits=[split],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[chat_math_metric(n)],
        version=0,
    )


TASKS_TABLE = [
    _math_cfg("aime24_chat", "HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a", N_AIME,
              lambda l: _aime_gold(l["answer"]), lambda l: {"id": l["id"]}),
    _math_cfg("aime25_chat", "yentinglin/aime_2025", "6f71d77b0b89b9dabe07ab466c51df33f514df7f", N_AIME,
              lambda l: _aime_gold(l["answer"]), lambda l: {"id": l["id"]}),
    _math_cfg("aime26_chat", "MathArena/aime_2026", "d61f486805306e6e6b5913504c07763a99e1b112", N_AIME,
              lambda l: _aime_gold(l["answer"]), lambda l: {"problem_idx": l["problem_idx"]}),
    _math_cfg("hmmt_feb_2026_chat", "MathArena/hmmt_feb_2026", "589a53fe152febc0f4f99f3e561007bf8da8e664", N_AIME,
              lambda l: _hmmt_gold(l["answer"]), lambda l: {"problem_idx": l["problem_idx"]}),
    # gold = the dataset's `solution` (its last \boxed is the answer), exactly the gold text of the base twin math_500
    _math_cfg("math_500_chat", "HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", N_MATH500,
              lambda l: l["solution"], lambda l: {"subject": l["subject"], "level": l["level"], "unique_id": l["unique_id"]}, split="test"),
]


# ------------------------------------------------------------------------------------------- multiple choice
MC_INSTRUCTION = (
    "Answer the following multiple choice question. The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering."
)


def mc_query(question: str, options: list[str]) -> str:
    letters = ascii_uppercase[: len(options)]
    body = "\n".join(f"{l}) {str(o).strip()}" for l, o in zip(letters, options))
    return f"{MC_INSTRUCTION.format(letters=letters)}\n\n{question.strip()}\n\n{body}"


def _gpqa_choices(line):
    """gpqa.py `_stable_choices`: [correct, wrong 1-3] shuffled by the md5 of the question (the same order in every run and for every model)."""
    items = [line["Correct Answer"], line["Incorrect Answer 1"], line["Incorrect Answer 2"], line["Incorrect Answer 3"]]
    seed = int(hashlib.md5(line["Question"].encode()).hexdigest(), 16) % (2**32)
    random.Random(seed).shuffle(items)
    return items, items.index(line["Correct Answer"])


def gpqa_chat_prompt(line, task_name: str = None):
    options, gold = _gpqa_choices(line)
    return Doc(
        task_name=task_name,
        query=mc_query(line["Question"], options),
        choices=list(ascii_uppercase[: len(options)]),
        gold_index=gold,
        specific={"domain": line.get("High-level domain"), "subdomain": line.get("Subdomain"), "record_id": line.get("Record ID")},
    )


def mmlu_pro_chat_prompt(line, task_name: str = None):
    options = list(line["options"])
    return Doc(
        task_name=task_name,
        query=mc_query(line["question"], options),
        choices=list(ascii_uppercase[: len(options)]),
        gold_index=int(line["answer_index"]),
        specific={"category": line["category"], "question_id": line["question_id"]},
    )


def mmlu_redux_chat_prompt(line, task_name: str = None, *, subject: str):
    if line["error_type"] != "ok":
        return None                     # the same rows as mmlu_redux:mcf_em
    options = list(line["choices"])
    gold = ascii_uppercase.index(line["answer"]) if isinstance(line["answer"], str) else int(line["answer"])
    return Doc(
        task_name=task_name,
        query=mc_query(line["question"], options),
        choices=list(ascii_uppercase[: len(options)]),
        gold_index=gold,
        specific={"subject": subject},
    )


_MMLU_SUBSETS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name="gpqa_diamond_chat",
        prompt_function=gpqa_chat_prompt,
        hf_repo="Idavidrein/gpqa",
        hf_subset="gpqa_diamond",
        hf_revision="83022cefff930aea54f654c0b282e74b9eeda5c6",
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[chat_choice_metric(N_GPQA)],
        version=0,
    ),
    LightevalTaskConfig(
        name="mmlu_pro_chat",
        prompt_function=mmlu_pro_chat_prompt,
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[chat_choice_metric(1)],
        version=0,
    ),
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"mmlu_redux_chat:{subset}",
        prompt_function=partial(mmlu_redux_chat_prompt, subject=subset),
        hf_repo="edinburgh-dawg/mmlu-redux-2.0",
        hf_subset=subset,
        hf_revision="372ea425445d51e1ba1188c56e5e893f8138621f",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[chat_choice_metric(1)],
        version=0,
    )
    for subset in _MMLU_SUBSETS
]


# ------------------------------------------------------------------------------------ instruction following
def _if_doc(line, task_name: str, kwargs_key: str = "kwargs"):
    return Doc(
        task_name=task_name,
        query=line["prompt"],           # no `instruction`: the prompt manager must not re-attach or strip anything
        choices=[""],
        gold_index=0,
        specific={"instruction_id_list": line["instruction_id_list"], "kwargs": line[kwargs_key]},
    )


def ifeval_chat_prompt(line, task_name: str = ""):
    d = _if_doc(line, task_name)
    d.specific = {"instructions_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]}     # the key spelling of ifeval's own metric
    return d


def ifbench_chat_prompt(line, task_name: str = ""):
    return _if_doc(line, task_name)       # IFBench's metric reads `instruction_id_list`


class _DeterministicChecks(SampleLevelComputation):
    """The base IFEval / IFBench metric, made safe to run for hours on an offline node (chat_core_checks.txt, 11-13). Everything else is the inner metric (same keys, same values):
    1. `langdetect` is unseeded (the language checkers of ~95 IFEval prompts could flip between two runs on the same completions): seeded before every sample;
       IFEval's letter_frequency checker picks a RANDOM letter when the prompt's letter is not alphabetic ('#', '!': 2 prompts): `random` is seeded per sample and restored.
    2. IFBench's checkers call nltk.download (once per process in Instruction.__init__, and for 'stopwords' on EVERY stop-word check); offline that is a network attempt per
       call: when the package is already on disk (nltk_data) the download is skipped, and the spaCy model / punkt flag is set when present.
    3. A checker that raises (IFBench: IndexError in format:quote_unquote / format:emoji, ZeroDivisionError in ratio:overlap on odd answers: 72 of 3,300 probes in the checks) would
       stop the whole lighteval call, hours of generations included: the sample scores 0 on every instruction and `checker_error` = 1 (logged; the share is a metric)."""

    def __init__(self, inner):
        self.inner = inner

    def compute(self, doc, model_response, **kwargs):
        import random

        from langdetect import DetectorFactory

        DetectorFactory.seed = 0
        _skip_downloads_when_present()
        state = random.getstate()
        random.seed(0)
        try:
            out = dict(self.inner.compute(doc, model_response, **kwargs))
            out["checker_error"] = 0
            return out
        except Exception as error:      # noqa: BLE001 (reward_checks/README.md item 5: catch every exception per answer, score 0, count it)
            spec = doc.specific or {}
            ids = spec.get("instruction_id_list") or spec.get("instructions_id_list") or []
            logger.warning(f"instruction checker raised {type(error).__name__}: {error} (ids {ids}); the sample scores 0 and counts in checker_error")
            k = len(ids)
            return {"prompt_level_strict_acc": 0, "inst_level_strict_acc": [False] * k, "prompt_level_loose_acc": 0, "inst_level_loose_acc": [False] * k, "checker_error": 1}
        finally:
            random.setstate(state)


def _nltk_has(package: str) -> bool:
    import os

    import nltk.data

    return any(os.path.exists(os.path.join(base, sub, name))
               for base in nltk.data.path for sub in ("corpora", "tokenizers", "taggers", "chunkers", "misc", "stemmers") for name in (package, package + ".zip"))


def _skip_downloads_when_present() -> None:
    try:
        import nltk
        import spacy.util
        from lighteval.tasks.tasks.ifbench import instructions as ifb

        if not getattr(nltk.download, "_local_first", False):
            real = nltk.download

            def local_first(info_or_id=None, *args, **kwargs):
                if isinstance(info_or_id, str) and _nltk_has(info_or_id):
                    return True
                return real(info_or_id, *args, **kwargs)

            local_first._local_first = True
            nltk.download = local_first
        if not ifb.RESOURCES_DOWNLOADED and _nltk_has("punkt_tab") and _nltk_has("averaged_perceptron_tagger_eng") and spacy.util.is_package("en_core_web_sm"):
            ifb.RESOURCES_DOWNLOADED = True
    except Exception:       # something is missing: leave the upstream behaviour (download) alone
        pass


def _deterministic(metric):
    import dataclasses

    import numpy as np

    return dataclasses.replace(
        metric,
        metric_name=[*metric.metric_name, "checker_error"],
        sample_level_fn=_DeterministicChecks(metric.sample_level_fn),
        corpus_level_fn={**metric.corpus_level_fn, "checker_error": np.mean},
        higher_is_better={**metric.higher_is_better, "checker_error": False},
    )


def _if_tasks():
    from lighteval.tasks.tasks.ifbench.main import ifbench_metrics
    from lighteval.tasks.tasks.ifeval.main import ifeval_metrics

    return [
        LightevalTaskConfig(
            name="ifeval_chat",
            prompt_function=ifeval_chat_prompt,
            hf_repo="google/IFEval",
            hf_subset="default",
            hf_revision="966cd89545d6b6acfd7638bc708b98261ca58e84",
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=BUDGET,
            stop_sequence=[],
            metrics=[_deterministic(ifeval_metrics)],
            version=0,
        ),
        LightevalTaskConfig(
            name="ifbench_chat",
            prompt_function=ifbench_chat_prompt,
            hf_repo="allenai/IFBench_test",
            hf_subset="default",
            hf_revision="ed7c12452594b7028b15807b86d3938150b663ae",
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=BUDGET,
            stop_sequence=[],
            metrics=[_deterministic(ifbench_metrics)],
            version=0,
        ),
    ]


TASKS_TABLE += _if_tasks()
