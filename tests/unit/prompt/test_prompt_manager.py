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

import random

import pytest

from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.prompt_manager import FewShotSampler
from lighteval.tasks.requests import Doc


@pytest.mark.parametrize("fewshot_select", ["sequential", "random", "balanced"])
def test_fewshot_sampler(fewshot_select: str):
    config = LightevalTaskConfig(
        name="test_fewshot_task",
        prompt_function=lambda _, __: None,
        hf_repo="",
        hf_subset="default",
        metrics=[],
        few_shots_split="test",
        few_shots_select=fewshot_select,
    )
    task = LightevalTask(config)
    rnd = random.Random(0)
    task._fewshot_docs = [
        Doc(str(i), ["A", "B"], rnd.randint(0, 2), fewshot_sorting_class=str(i % 20)) for i in range(100)
    ]
    sampler = FewShotSampler(task)
    seed = 1
    docs = sampler.sample_fewshot_examples(20, seed)

    match task.fewshot_selection:
        case "sequential":
            assert docs == task.fewshot_docs()[:20]
        case "random":
            rnd = random.Random(seed)
            task_docs = task.fewshot_docs()
            rnd.shuffle(task_docs)
            assert docs == task_docs[:20]


@pytest.mark.parametrize("pool_size", [0, 1, 4, 5, 6, 20])
@pytest.mark.parametrize("eval_in_pool", [False, True])
def test_random_fewshot_sampler_small_pools(pool_size: int, eval_in_pool: bool):
    config = LightevalTaskConfig(
        name="test_fewshot_task",
        prompt_function=lambda _, __: None,
        hf_repo="",
        hf_subset="default",
        metrics=[],
        few_shots_split="test",
        few_shots_select="random_sampling",
    )
    task = LightevalTask(config)
    task._fewshot_docs = [Doc(str(i), ["A", "B"], 0) for i in range(pool_size)]
    eval_doc = Doc("0" if eval_in_pool else "evaluation", ["A", "B"], 0)
    sampler = FewShotSampler(task)

    samples = sampler.sample_fewshot_examples(5, 0, sampler=random.Random(42), formatted_doc=eval_doc)

    eligible = [doc for doc in task._fewshot_docs if doc != eval_doc]
    assert len(samples) == min(5, len(eligible))
    assert eval_doc not in samples
    assert all(doc in eligible for doc in samples)
    assert len({doc.query for doc in samples}) == len(samples)
    assert samples == sampler.sample_fewshot_examples(5, 0, sampler=random.Random(42), formatted_doc=eval_doc)

    # Preserve the existing seeded draw for pools that already have enough candidates.
    if pool_size >= 6:
        candidates = random.Random(42).sample(task._fewshot_docs, 6)
        assert samples == [doc for doc in candidates if doc != eval_doc][:5]


@pytest.mark.parametrize("pool_size", [1, 3, 4, 6, 20])
@pytest.mark.parametrize("fewshot_select", ["random_sampling", "balanced", "sequential", "random"])
def test_get_docs_excludes_current_question(pool_size: int, fewshot_select: str):
    config = LightevalTaskConfig(
        name="test_fewshot_task",
        prompt_function=lambda row, task_name: Doc(row["question"], ["A", "B"], 0, task_name=task_name),
        hf_repo="",
        hf_subset="default",
        metrics=[],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select=fewshot_select,
        num_fewshots=5,
    )
    task = LightevalTask(config)
    task.dataset = {"test": [{"question": str(i)} for i in range(pool_size)]}

    # Repeated preparation also changes metadata such as fewshot_samples.
    for _ in range(2):
        docs = task.get_docs()
        for doc in docs:
            assert len(doc.fewshot_samples) == min(5, pool_size - 1)
            assert all(sample.query != doc.query for sample in doc.fewshot_samples)


def test_fewshot_exclusion_preserves_different_choices():
    config = LightevalTaskConfig(
        name="test_fewshot_task",
        prompt_function=lambda _, __: None,
        hf_repo="",
        hf_subset="default",
        metrics=[],
        few_shots_split="test",
        few_shots_select="random_sampling",
    )
    task = LightevalTask(config)
    candidate = Doc("Pick the correct option", ["A", "B"], 0)
    task._fewshot_docs = [candidate]
    eval_doc = Doc("Pick the correct option", ["C", "D"], 0)
    assert task.fewshot_sampler.sample_fewshot_examples(
        1, 0, sampler=random.Random(42), formatted_doc=eval_doc
    ) == [candidate]
