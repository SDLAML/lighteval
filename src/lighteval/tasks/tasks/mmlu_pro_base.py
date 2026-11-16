"""
name:
MMLU-Pro, base-model protocol (5-shot chain of thought, the official TIGER-Lab prompt)

dataset:
TIGER-Lab/MMLU-Pro

abstract:
The official evaluation of MMLU-Pro for base models (TIGER-AI-Lab/MMLU-Pro `evaluate_from_local.py` + `cot_prompt_lib/initial_prompt.txt`,
commit f418b116): the prompt is the one-line header "The following are multiple choice questions (with answers) about <category>. Think step
by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.", then the first 5 questions of the
VALIDATION split of the same category as worked examples (question, "Options:" with the lettered options, "Answer: Let's think step by step."
and the dataset's own `cot_content`, ending "The answer is (X)."), then the evaluated question in the same layout ending "Answer: Let's think
step by step.". Greedy decoding, at most 2048 new tokens, stopped at "Question:". The answer is read exactly as the official script reads it:
the first "answer is (X)" (letters A-J), else "answer: X", else the last stand-alone letter A-J of the text.
Differences from the official script (documented in docs/evals/tasks/mmlu_pro_base_cot.md): (1) an output with NO letter at all scores 0 (the script
guesses a random option); (2) the script drops shots when the prompt would exceed 2048 tokens (its 4096-token context); here the context is 8192 and
all 5 shots are always shown; (3) options equal to "N/A" are dropped as in the script (none occur in the test split); (4) the gold is the `answer` letter, as the script compares it (one question,
3983, has `answer_index` 1 for `answer` "C"). One pooled task over all 12,032
test questions: the score is the overall accuracy (the official "total"), as the paper and the Qwen3 / Qwen2.5 base reports use it; the category is
kept in `doc.specific` for a breakdown from the details. Metrics: acc, answer_is_rate (share of outputs with the first-level "answer is (X)").

languages:
english

tags:
general-knowledge, knowledge, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2406.01574
"""

import re
from string import ascii_uppercase
from typing import Any

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod


REPO = "TIGER-Lab/MMLU-Pro"
REVISION = "3373e0b32277875b8db2aa555a333b78a08477ea"  # the pin of mmlu_pro_chat too (card license: MIT)
MAX_NEW_TOKENS = 2048  # official: max_new_tokens
STOP_SEQUENCES = ["Question:"]  # official: stop=["Question:"]
TASK_VERSION = 0

# cot_prompt_lib/initial_prompt.txt is "<line>\n\n\n" and the script adds "\n" before the first example: four newlines. lighteval joins the
# instruction, every shot and the question with "\n\n", so the instruction itself ends with two newlines.
HEADER = (
    "The following are multiple choice questions (with answers) about {category}. Think step by step and then finish your answer with "
    '"the answer is (X)" where X is the correct letter choice.\n\n'
)
_COT_PREFIX = "A: Let's think step by step."


def _options(line) -> list[str]:
    """The options without "N/A" padding (official `preprocess`; the padding is always at the end, so the letters of the others do not move)."""
    return [option for option in line["options"] if option != "N/A"]


def _question_block(line) -> str:
    """"Question:\\n<question>\\nOptions:\\nA. <option>\\n...\\nAnswer:" (official format_cot_example, without the answer)."""
    options = "".join(f"{letter}. {option}\n" for letter, option in zip(ascii_uppercase, _options(line)))
    return f"Question:\n{line['question']}\nOptions:\n{options}Answer:"


def mmlu_pro_base_prompt(line, task_name: str = None):
    category = line["category"]
    if line.get("__few_shots"):
        cot = line["cot_content"]
        if not cot.startswith(_COT_PREFIX):
            raise ValueError(f"MMLU-Pro validation question {line['question_id']}: cot_content does not start with {_COT_PREFIX!r}")
        # the shot reads "<block> Let's think step by step. <reasoning> The answer is (X)." (lighteval joins the query and the gold with one space)
        return Doc(
            task_name=task_name,
            query=_question_block(line),
            choices=[cot[len("A: "):]],
            gold_index=0,
            specific={"fewshot_group": category},
        )
    letters = list(ascii_uppercase[: len(_options(line))])
    if line["answer"] not in letters:
        raise ValueError(f"MMLU-Pro question {line['question_id']}: gold {line['answer']!r} is not one of the options {letters}")
    return Doc(
        task_name=task_name,
        query=f"{_question_block(line)} Let's think step by step.",
        choices=letters,
        gold_index=letters.index(line["answer"]),     # the letter, as the official script compares it (question 3983 has answer_index 1 for answer "C": the script scores by `answer`)
        instruction=HEADER.format(category=category),
        specific={"category": category, "fewshot_group": category, "question_id": int(line["question_id"]), "n_options": len(letters)},
    )


# The official extraction (evaluate_from_local.py extract_answer / extract_again / extract_final), line for line.
_FIRST = re.compile(r"answer is \(?([A-J])\)?")
_SECOND = re.compile(r".*[aA]nswer:\s*([A-J])")
_THIRD = re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", re.DOTALL)


def extract_answer(text: str) -> tuple[str | None, int]:
    """(letter or None, level 1-3 of the rule that found it; 0 when none did)."""
    text = text or ""
    for level, pattern in ((1, _FIRST), (2, _SECOND), (3, _THIRD)):
        match = pattern.search(text)
        if match:
            return (match.group(1) if level < 3 else match.group(0)), level
    return None, 0


class OfficialExtraction(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> dict[str, Any]:
        text = (model_response.final_text[0] if model_response.final_text else "") or ""
        letter, level = extract_answer(text)
        return {"acc": float(letter is not None and letter == doc.choices[doc.gold_index]), "answer_is_rate": float(level == 1)}


MMLU_PRO_BASE_METRIC = SampleLevelMetricGrouping(
    metric_name=["acc", "answer_is_rate"],
    sample_level_fn=OfficialExtraction(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={"acc": np.mean, "answer_is_rate": np.mean},
    higher_is_better={"acc": True, "answer_is_rate": True},
)

TASKS_TABLE = [
    LightevalTaskConfig(
        name="mmlu_pro_base:cot",
        prompt_function=mmlu_pro_base_prompt,
        hf_repo=REPO,
        hf_subset="default",
        hf_revision=REVISION,
        hf_avail_splits=["test", "validation"],
        evaluation_splits=["test"],
        few_shots_split="validation",  # 70 questions, 5 per category
        few_shots_select="sequential",  # the first K of the evaluated question's category, in dataset order (doc.specific["fewshot_group"])
        generation_size=MAX_NEW_TOKENS,
        metrics=[MMLU_PRO_BASE_METRIC],
        stop_sequence=STOP_SEQUENCES,
        version=TASK_VERSION,
    )
]
