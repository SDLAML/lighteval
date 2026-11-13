"""
name:
Drop Qa

dataset:
lighteval/drop_harness

abstract:
The DROP dataset is a new question-answering dataset designed to evaluate the
ability of language models to answer complex questions that require reasoning
over multiple sentences.

languages:
english

tags:
math, qa, reasoning

paper:
https://arxiv.org/abs/1810.00505
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def drop_gen_prompt(line, task_name: str = None):
    """GenQA variant: CF-style prompt, generate answer, score with the DROP EM / F1 (max over all reference answers).

    References = the main answer + every `validated_answers` entry, de-duplicated (ported from upstream lighteval
    `drop_qa.py` `drop_prompt`; same as lm-evaluation-harness `get_answers` and OLMES drop.py). The official DROP
    evaluator and the harness take the max over all of them. A reference with several spans is shown joined by ", ".
    """

    def _flatten_validated_answers(validated_answers):
        valid_answers = []
        for i in range(len(validated_answers["number"])):
            valid_answers.append(
                {
                    "number": validated_answers["number"][i],
                    "date": validated_answers["date"][i],
                    "spans": validated_answers["spans"][i],
                }
            )
        return valid_answers

    def parse_answer(answer):
        if answer["number"] != "":
            return (str(answer["number"]),)
        if answer["spans"] != []:
            return tuple(answer["spans"])
        return (" ".join([answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]]).strip(),)

    answers = []
    answers_set = set()
    candidates = [line["answer"]] + _flatten_validated_answers(line["validated_answers"])
    for candidate in candidates:
        answer = parse_answer(candidate)
        if answer in answers_set:
            continue
        answers.append(answer)
        answers_set.add(answer)

    is_few_shots = line.get("__few_shots", False)
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {line['question']}\nAnswer:",
        choices=[f"{' ' if is_few_shots else ''}{', '.join(a)}" for a in answers],
        gold_index=list(range(len(answers))),
        specific={"golds_no_preprocessing": [list(a) for a in answers]},
    )


def drop_bpb_prompt(line, task_name: str = None):
    """BPB variant: CF-style prompt with only the first gold answer."""
    answer = line["answer"]
    if answer["number"] != "":
        gold_text = str(answer["number"])
    elif answer["spans"]:
        gold_text = ", ".join(answer["spans"])
    else:
        gold_text = " ".join([answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]]).strip()
    if not gold_text:
        return None  # degenerate sample with no extractable answer; skip
    if not gold_text[0].isspace():
        gold_text = " " + gold_text
    return Doc(
        task_name=task_name,
        query=f"Passage: {line['passage']}\nQuestion: {line['question']}\nAnswer:",
        choices=[gold_text],
        gold_index=0,
    )

# lighteval/drop_harness: the snapshot of the reference runs (same revision as configs/datasets.lock.yaml in titan_eval)
_REVISION = "9bab3c8c1b5241e83b9e2350b1c9f3940c86fc53"


drop_bpb = LightevalTaskConfig(
    name="drop:bpb",
    prompt_function=drop_bpb_prompt,
    hf_repo="lighteval/drop_harness",
    hf_revision=_REVISION,
    hf_subset="default",
    evaluation_splits=("validation",),
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=-1,
    stop_sequence=["\n\n", "Passage:", "Question:"],
    metrics=[Metrics.target_bits_per_byte],
    version=0,
)

drop_gen = LightevalTaskConfig(
    name="drop:gen",
    prompt_function=drop_gen_prompt,
    hf_repo="lighteval/drop_harness",
    hf_revision=_REVISION,
    hf_subset="default",
    evaluation_splits=("validation",),
    few_shots_split="train",
    few_shots_select="random_sampling_from_train",
    generation_size=100,
    stop_sequence=["\n\n", "Passage:", "Question:"],
    metrics=[Metrics.drop],
    version=1,
)

TASKS_TABLE = [
    drop_bpb,
    drop_gen,
]
