import ast

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

def popqa_prompt(line, task_name: str = None):
    return Doc(task_name=task_name, 
               query=f"{line['question']} ", 
               gold_index=0, 
               choices=[ast.literal_eval(line["possible_answers"])]
               )

popqa = LightevalTaskConfig(
    name="popqa",
    prompt_function=popqa_prompt,
    hf_repo="akariasai/PopQA",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=8,
    metrics=[Metrics.exact_match],
    stop_sequence=["\n"],
    version=0,
)

TASKS_TABLE = [
    popqa,
]