"""
name:
KINA (Knowledge Index of Noah's Ark)

dataset:
2077AIDataFoundation/KINA

abstract:
A multidisciplinary knowledge benchmark with 899 graduate-level questions,
each with ten answer options (A-J). Only the question and candidate answers
are shown to the model; option-level explanations are excluded. The benchmark
has a single test split, so standard evaluation uses zero shots. Subtasks use
the root discipline in each hierarchical path, such as Engineering or Law.

languages:
english

tags:
graduate-level, knowledge, multiple-choice, qa, reasoning

paper:
https://arxiv.org/abs/2606.05104
"""

from dataclasses import replace
from functools import partial

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric, MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


_LABELS = "ABCDEFGHIJ"
_KINA_DISCIPLINES = (
    "Agronomy",
    "Economics",
    "Education",
    "Engineering",
    "History",
    "Law",
    "Literature and Arts",
    "Management",
    "Medicine",
    "Philosophy",
    "Science",
    "Sociology",
)
# Follow the reference evaluation's reasoning prompt and letter extraction:
# https://github.com/2077AI/KINA/blob/main/src/kina_bench/utils.py
_COT_INSTRUCTION = (
    "Answer the following multiple choice question. There is only one correct answer. "
    "The last line of your response should be in the format 'Answer: $LETTER' (without quotes), "
    "where LETTER is one of A, B, C, D, E, F, G, H, I, or J. Think step by step before answering."
)
_INSPECT_TEMPLATE = (
    "Answer the following multiple choice question.\n\nQuestion:\n{question}\n\nOptions:\n{choices}\n\n"
    "Reply with 'ANSWER: $LETTER' (without quotes), where LETTER is the correct option."
)
_INSPECT_COT_TEMPLATE = _COT_INSTRUCTION.replace("Answer:", "ANSWER:") + "\nQuestion:\n{question}\nOptions:\n{choices}"


def _matches_discipline(record, discipline: str):
    return record["discipline"].split("/", 1)[0].strip() == discipline


def _matches_sample_discipline(sample: Sample, discipline: str):
    return _matches_discipline(sample.metadata, discipline)


def _option_texts(record):
    """Order answers by their explicit keys without exposing explanations."""
    options = {option["key"]: option["answer"].strip() for option in record["options"]}
    return [options[label] for label in _LABELS]


def kina_prompt(line, task_name: str = None, *, formulation: str = "mcf"):
    question = line["question"].strip()
    options = _option_texts(line)
    instruction = ""
    if formulation == "cf":
        query = f"Question: {question}\nAnswer:"
        choices = [f" {option}" for option in options]
    else:
        option_text = "\n".join(f"{label}: {option}" for label, option in zip(_LABELS, options))
        choices = [f" {label}" for label in _LABELS]
        if formulation == "cot":
            instruction = _COT_INSTRUCTION
            query = f"{instruction}\nQuestion:\n{question}\nOptions:\n{option_text}"
        else:
            query = f"Answer the following multiple choice question.\n\nQuestion: {question}\n{option_text}\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=_LABELS.index(line["correct_answer"].strip()),
        instruction=instruction,
        specific={"index": line["index"], "discipline": line["discipline"]},
    )


def record_to_sample(record):
    return Sample(
        id=record["index"],
        input=record["question"].strip(),
        choices=_option_texts(record),
        target=record["correct_answer"].strip(),
        metadata={"discipline": record["discipline"]},
    )


def sample_to_fewshot(sample):
    options = "\n".join(f"{label}) {option}" for label, option in zip(_LABELS, sample.choices))
    return f"{sample.input}\n\n{options}\n\nANSWER: {sample.target}"


_LIKELIHOOD_METRICS = [LogLikelihoodAccMetric(), LogLikelihoodAccMetric(normalization=LogProbCharNorm())]

# Require an answer anchor or a standalone label at the start of a line.
# The unrestricted GPQA fallback can mistake the "A" in an empty "Answer:" for a prediction.
_COT_METRIC = replace(
    Metrics.gpqa_instruct_metric.value,
    sample_level_fn=MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        gold_extraction_target=[
            IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=True)
        ],
        pred_extraction_target=[
            IndicesExtractionConfig(prefix_for_extraction="NativeLetters", try_extract_without_anchor=False)
        ],
    ),
)

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"kina:{discipline.lower().replace(' ', '_')}:{variant}",
        prompt_function=partial(kina_prompt, formulation=variant),
        hf_repo="2077AIDataFoundation/KINA",
        hf_subset="default",
        hf_filter=partial(_matches_discipline, discipline=discipline),
        filter=partial(_matches_sample_discipline, discipline=discipline),
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        # There is no dedicated few-shot split; use |0 for standard evaluation.
        few_shots_split="test",
        few_shots_select="random_sampling",
        sample_fields=record_to_sample,
        sample_to_fewshot=sample_to_fewshot,
        solver=[
            multiple_choice(
                template=_INSPECT_COT_TEMPLATE if variant == "cot" else _INSPECT_TEMPLATE,
                cot=variant == "cot",
                cache=True,
            )
        ],
        scorer=choice(),
        generation_size=generation_size,
        metrics=metrics,
        stop_sequence=[] if variant == "cot" else ["\n"],
        version=0,
    )
    for discipline in _KINA_DISCIPLINES
    for variant, metrics, generation_size in [
        ("cf", [*_LIKELIHOOD_METRICS, Metrics.target_bits_per_byte], -1),
        ("mcf", _LIKELIHOOD_METRICS, -1),
        ("mcf_em", [Metrics.exact_match], 5),
        ("cot", [_COT_METRIC], 32768),
    ]
]
