from importlib import import_module

import pytest

from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.prompt_manager import FewShotSampler
from lighteval.tasks.requests import Doc


FORMATTING_CASES = [
    ("simpleqa", "simpleqa_gen_prompt", {"problem": "Question?", "answer": "Answer"}),
    ("triviaqa", "triviaqa_gen_prompt", {"question": "Question?", "answer": {"value": "Answer", "aliases": []}}),
    ("natural_questions", "nq_gen_prompt", {"question": "Question?", "answer": ["Answer"]}),
    ("jeopardy", "jeopardy_gen_prompt", {"context": "Category: Question?", "continuation": "Answer"}),
    ("squad", "squad_gen_prompt", {
        "title": "Title", "context": "Context", "question": "Question?", "answers": {"text": ["Answer"]},
    }),
    ("drop_qa", "drop_gen_prompt", {
        "passage": "Passage", "question": "Question?", "answer": {"number": "1", "spans": []},
    }),
    ("coqa", "coqa_gen_prompt", {
        "story": "Story", "questions": {"input_text": ["First?", "Second?"]},
        "answers": {"input_text": ["First answer", "Second answer"]},
    }),
    ("gsm8k", "gsm8k_prompt", {"question": "One plus one?", "answer": "One plus one is two. #### 2"}),
    ("gsm_symbolic", "gsm_symbolic_prompt", {"question": "One plus one?", "answer": "One plus one is two. #### 2"}),
    ("bigbench_hard", "bbh_mcf_prompt", {"input": "Question?\nOptions:\n(A) First\n(B) Second", "target": "(B)"}),
    ("bigbench_hard", "bbh_chat_prompt", {"input": "Question?\nOptions:\n(A) First\n(B) Second", "target": "(B)"}),
]


@pytest.mark.parametrize("module_name,prompt_name,row", FORMATTING_CASES)
@pytest.mark.parametrize("strategy", ["random_sampling", "random_sampling_from_train", "balanced", "random", "sequential"])
def test_fewshot_formatting_cannot_leak_source_row(module_name, prompt_name, row, strategy):
    prompt = getattr(import_module(f"lighteval.tasks.tasks.{module_name}"), prompt_name)
    config = LightevalTaskConfig(
        name="formatting_test", prompt_function=prompt, hf_repo="", hf_subset="default", metrics=[],
        evaluation_splits=["test"], few_shots_split="test", few_shots_select=strategy, num_fewshots=5,
    )
    task = LightevalTask(config)
    task.dataset = {"test": [dict(row)]}
    docs = task.get_docs()
    assert docs
    assert all(doc.source_id == ("test", 0) for doc in docs)
    assert all(doc.fewshot_samples == [] for doc in docs)


def test_source_identity_handles_different_question_and_answer_formatting():
    def prompt(row, task_name):
        if row["__few_shots"]:
            return Doc("Worked example: Question?", ["Reasoning and answer"], 0)
        return Doc("Question?", ["Answer"], 0)

    task = LightevalTask(LightevalTaskConfig(
        name="formatting_test", prompt_function=prompt, hf_repo="", hf_subset="default", metrics=[],
        evaluation_splits=["test"], few_shots_split="test", few_shots_select="random_sampling", num_fewshots=5,
    ))
    task.dataset = {"test": [{}]}
    assert task.get_docs()[0].fewshot_samples == []


def test_source_identity_does_not_confuse_row_numbers_in_different_splits():
    task = LightevalTask(LightevalTaskConfig(
        name="split_test",
        prompt_function=lambda row, _: Doc(row["question"], ["Answer"], 0),
        hf_repo="", hf_subset="default", metrics=[], evaluation_splits=["test"],
        few_shots_split="train", few_shots_select="random_sampling_from_train", num_fewshots=1,
    ))
    task.dataset = {"test": [{"question": "Evaluation?"}], "train": [{"question": "Training?"}]}
    doc = task.get_docs()[0]
    assert doc.source_id == ("test", 0)
    assert len(doc.fewshot_samples) == 1
    assert doc.fewshot_samples[0].source_id == ("train", 0)


def test_duplicate_content_exclusion_ignores_answer_whitespace():
    evaluation = Doc("Question?", ["Answer"], 0, source_id=("test", 0))
    duplicate = Doc("Question?", [" Answer "], 0, source_id=("train", 10))
    assert FewShotSampler._is_same_example(duplicate, evaluation)


def test_same_row_exclusion_includes_other_turns():
    evaluation = Doc("First question?", ["First answer"], 0, source_id=("test", 0))
    later_turn = Doc("First question? First answer. Second question?", ["Second answer"], 0, source_id=("test", 0))
    assert FewShotSampler._is_same_example(later_turn, evaluation)
