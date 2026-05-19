"""
name:
Winogrande

dataset:
allenai/winogrande

abstract:
WinoGrande is a new collection of 44k problems, inspired by Winograd Schema
Challenge (Levesque, Davis, and Morgenstern 2011), but adjusted to improve the
scale and robustness against the dataset-specific bias. Formulated as a
fill-in-a-blank task with binary options, the goal is to choose the right option
for a given sentence which requires commonsense reasoning.

languages:
english

tags:
commonsense, multiple-choice

paper:
https://arxiv.org/abs/1907.10641
"""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def winogrande_cf_prompt(line, task_name: str = None):
    """Partial evaluation (Trinh & Le 2018), matching OLMO's implementation.

    Context = sentence with the gold option substituted in (prefix + option).
    Gold = the sentence suffix after the blank, with a leading space.
    BPB = logprob(suffix | prefix + gold_option) / bytes(suffix).

    Accuracy is not meaningful in this single-choice format (always 1.0), so
    only BPB is reported. Use winogrande:cf for accuracy scoring.
    """
    sentence = line["sentence"]
    blank_pos = sentence.index("_")
    prefix = sentence[:blank_pos].rstrip()
    suffix = sentence[blank_pos + 1:].strip()

    gold_ix = int(line["answer"]) - 1 if line["answer"].strip() else -1
    gold_option = [line["option1"], line["option2"]][gold_ix]

    return Doc(
        task_name=task_name,
        query=prefix + " " + gold_option,
        choices=[" " + suffix],
        gold_index=0,
    )


def winogrande_mcf_prompt(line, task_name: str = None):
    """MCF variant: Fill-in-the-blank format matching OLMO's WinograndeMC.

    Replaces _ with ___ in the sentence, lists options as A/B without the suffix.
    """
    sentence_with_blank = line["sentence"].replace("_", "___")
    query = f"Fill in the blank: {sentence_with_blank}\n A. {line['option1']}\n B. {line['option2']}\nAnswer:"
    gold_ix = int(line["answer"]) - 1 if line["answer"] != "" else -1
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B"],
        gold_index=gold_ix,
    )


def winogrande_rc_prompt(line, task_name: str = None):
    """RC (rank choice) variant: standard CF approach for accuracy.

    Context = sentence prefix before blank.
    Choices = full sentence completions with each option substituted in.
    Approximates OLMO's RC_none (partial eval) within lighteval's Doc framework:
    computes P(option+suffix | prefix) rather than P(suffix | prefix+option).
    """
    sentence = line["sentence"]
    blank_pos = sentence.index("_")
    prefix = sentence[:blank_pos].rstrip()
    suffix = " " + sentence[blank_pos + 1:].strip()
    gold_ix = int(line["answer"]) - 1 if line["answer"].strip() else -1
    return Doc(
        task_name=task_name,
        query=prefix,
        choices=[
            " " + line["option1"] + suffix,
            " " + line["option2"] + suffix,
        ],
        gold_index=gold_ix,
    )


# BPB variant: partial evaluation (OLMO-style), BPB only
# Accuracy is not reported because partial eval uses one fixed gold choice.
winogrande_bpb = LightevalTaskConfig(
    name="winogrande:bpb",
    prompt_function=winogrande_cf_prompt,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=[Metrics.target_bits_per_byte],
    stop_sequence=["\n"],
    version=0,
)

# MCF variant: "Fill in the blank:" format matching OLMO's WinograndeMC
winogrande_mcf = LightevalTaskConfig(
    name="winogrande:mcf",
    prompt_function=winogrande_mcf_prompt,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# Greedy variant: MCF-style prompt, generate 1 token, exact match
winogrande_mcf_em = LightevalTaskConfig(
    name="winogrande:mcf_em",
    prompt_function=winogrande_mcf_prompt,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

# CF variant: standard cloze-form for accuracy (approximates Table 46 RC_none).
# Scores P(option+suffix | prefix) rather than OLMO's P(suffix | prefix+option),
# but gives meaningful accuracy using lighteval's standard Doc framework.
winogrande_cf = LightevalTaskConfig(
    name="winogrande:cf",
    prompt_function=winogrande_rc_prompt,
    hf_repo="allenai/winogrande",
    hf_subset="winogrande_xl",
    hf_avail_splits=["train", "test", "validation"],
    evaluation_splits=["validation"],
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=[
        LogLikelihoodAccMetric(),
    ],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    winogrande_bpb,
    winogrande_cf,
    winogrande_mcf,
    winogrande_mcf_em,
]
