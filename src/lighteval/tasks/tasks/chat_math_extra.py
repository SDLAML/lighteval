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
Chat math extras (titan_eval, post-trained models): GSM8K, OlympiadBench (English text math), HMMT Feb 2025, Fusion Aya Math (4 languages)

dataset:
openai/gsm8k, Hothan/OlympiadBench, MathArena/hmmt_feb_2025, tiny-aya-math-edition/fusion-aya-math-bench

abstract:
Chat twins of the math tasks, same protocol as math_500_chat / aime24_chat (tasks/chat_evals.py): the chat template and thinking switch
come from the model's settings, 0-shot instruction prompt with the boxed-answer format, the scored text is what follows the LAST </think>
(an unclosed block gives no answer), `\\boxed{}` strict and lenient numbers, pass@k over the samples (chat_metrics.chat_math_metric).
Samples per prompt: GSM8K 1, OlympiadBench 4, HMMT Feb 2025 32, Fusion Aya Math 4 (the task's default; the run's gen.n replaces it).
The gold of every math task is read by the same rules as the base task: GSM8K = the number after "####" (commas removed), Fusion Aya
Math = its `fusion_aya_gold` (bare comma lists and tuples written as $...$, see tasks/multilingual/tasks/fusion_aya_math.py).

languages:
english, german, french, ukrainian (fusion_aya_math_chat only; the instruction is English, the problem in its own language)

tags:
chat, reasoning, math

paper:
https://arxiv.org/abs/2505.09388
"""

from langcodes import standardize_tag

from lighteval.metrics.chat_metrics import chat_math_metric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks.fusion_aya_math import _LANGUAGES as FUSION_LANGUAGES
from lighteval.tasks.multilingual.tasks.fusion_aya_math import _REVISION as FUSION_REVISION
from lighteval.tasks.multilingual.tasks.fusion_aya_math import fusion_aya_gold
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.chat_evals import BUDGET, MATH_PROMPT, N_AIME, N_MATH500, _hmmt_gold


N_GSM8K = 1
N_OLYMPIAD = 4
N_FUSION = 4


def _math_chat_doc(problem: str, gold: str, task_name: str = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=MATH_PROMPT.format(problem=problem.strip()),
        choices=[gold],
        gold_index=0,
    )


def gsm8k_chat_prompt(line, task_name: str = None):
    # the gold = the number after the last '####', commas removed (tasks/tasks/gsm8k.py final_number, version 1)
    gold = line["answer"].split("####")[-1].strip().replace(",", "")
    return _math_chat_doc(line["question"], gold, task_name)


def olympiad_chat_prompt(line, task_name: str = None):
    # the base task (tasks/tasks/olympiade_bench.py) asks answer-type aware instructions; the chat twin uses the MATH instruction (as math_500_chat).
    # `final_answer` is a list of ONE string in all 674 rows (checked 11-14); multi-answer golds are already written as one string ("$69$,$84$").
    # bare comma lists / tuples ("1,3,5", "(6,5)") written as $...$ by the fusion rule (version 1 gold, 33 of 674 unscorable otherwise)
    return _math_chat_doc(line["question"], fusion_aya_gold(str(line["final_answer"][0]).strip()), task_name)


def hmmt_2025_chat_prompt(line, task_name: str = None):
    return _math_chat_doc(line["problem"], _hmmt_gold(line["answer"]), task_name)


def fusion_chat_prompt(line, task_name: str = None):
    return _math_chat_doc(line["question"], fusion_aya_gold(line["answer"]), task_name)


def _cfg(name, repo, subset, revision, split, prompt_function, n) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=name,
        prompt_function=prompt_function,
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
    _cfg("gsm8k_chat", "openai/gsm8k", "main", "740312add88f781978c0658806c59bc2815b9866", "test", gsm8k_chat_prompt, N_GSM8K),
    _cfg("olympiadbench_chat", "Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "91184b52131e7fc9455fef848035173aea8cc01a", "train",
         olympiad_chat_prompt, N_OLYMPIAD),
    _cfg("hmmt_feb_2025_chat", "MathArena/hmmt_feb_2025", "default", "6fdc4277120810ff75aa22d2d5489b91f7a262a1", "train",
         hmmt_2025_chat_prompt, N_AIME),
] + [
    _cfg(f"fusion_aya_math_chat:{language.value}", "tiny-aya-math-edition/fusion-aya-math-bench", standardize_tag(language.value),
         FUSION_REVISION, "benchmark", fusion_chat_prompt, N_FUSION)
    for language in FUSION_LANGUAGES
]
