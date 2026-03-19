"""
name:
Siqa

dataset:
allenai/social_i_qa

abstract:
We introduce Social IQa: Social Interaction QA, a new question-answering
benchmark for testing social commonsense intelligence. Contrary to many prior
benchmarks that focus on physical or taxonomic knowledge, Social IQa focuses on
reasoning about people's actions and their social implications. For example,
given an action like "Jesse saw a concert" and a question like "Why did Jesse do
this?", humans can easily infer that Jesse wanted "to see their favorite
performer" or "to enjoy the music", and not "to see what's happening inside" or
"to see if it works". The actions in Social IQa span a wide variety of social
situations, and answer candidates contain both human-curated answers and
adversarially-filtered machine-generated candidates. Social IQa contains over
37,000 QA pairs for evaluating models' abilities to reason about the social
implications of everyday events and situations.

languages:
english

tags:
commonsense, multiple-choice, qa

paper:
"""

from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def siqa_mcf_prompt(line, task_name: str = None):
    """MCF variant: labeled A/B/C options in prompt, score label tokens via logprobs."""
    query = "The following are multiple choice questions (with answers) about common sense.\n"
    query += f"Question: {line['context']} {line['question']}\n"
    query += "".join(
        [
            f"{key}. {choice}\n"
            for key, choice in zip(list(ascii_uppercase)[:3], [line["answerA"], line["answerB"], line["answerC"]])
        ]
    )
    query += "Answer: "

    return Doc(
        task_name=task_name,
        query=query,
        choices=["A", "B", "C"],
        gold_index=int(line["label"]) - 1,
        instruction="The following are multiple choice questions (with answers) about common sense.\n",
    )


def siqa_cf_prompt(line, task_name: str = None):
    """CF variant: completion-style prompt with full answer texts as choices."""
    answers = [line["answerA"], line["answerB"], line["answerC"]]
    query = f"Context: {line['context']}\nQuestion: {line['question']}\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" " + a for a in answers],
        gold_index=int(line["label"]) - 1,
    )


# Greedy variant: MCF-style prompt, generate 1 token, exact match
siqa_mcf_em = LightevalTaskConfig(
    name="siqa:mcf_em",
    prompt_function=siqa_mcf_prompt,
    hf_repo="lighteval/siqa",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=1,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

# MCF variant: labeled options, score label tokens via logprobs (TRUE MCF)
siqa_mcf = LightevalTaskConfig(
    name="siqa:mcf",
    prompt_function=siqa_mcf_prompt,
    hf_repo="lighteval/siqa",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_MCF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

# CF variant: completion-style, logprob on full answer text + BPB on gold choice
siqa_cf = LightevalTaskConfig(
    name="siqa:cf",
    prompt_function=siqa_cf_prompt,
    hf_repo="lighteval/siqa",
    hf_subset="default",
    hf_avail_splits=["train", "validation"],
    evaluation_splits=["validation"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    metrics=_CF_METRICS,
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    siqa_mcf_em,
    siqa_mcf,
    siqa_cf,
]
