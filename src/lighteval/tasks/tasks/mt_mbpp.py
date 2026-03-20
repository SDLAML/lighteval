"""
MT-MBPP BPB evaluation tasks for lighteval.

Dataset : allenai/multilingual_mbpp (HuggingFace), one config per language
ICL     : 3 shots sampled from the train split
Metric  : BPB = -log2 p(code | text) / bytes(code)

The `text` field contains the problem description / docstring.
The `code` field contains the reference implementation in the target language.

17 language subtasks:
  mt_mbpp:bash:bpb       mt_mbpp:c:bpb
  mt_mbpp:cpp:bpb        mt_mbpp:csharp:bpb
  mt_mbpp:go:bpb         mt_mbpp:haskell:bpb
  mt_mbpp:java:bpb       mt_mbpp:javascript:bpb
  mt_mbpp:matlab:bpb     mt_mbpp:php:bpb
  mt_mbpp:python:bpb     mt_mbpp:r:bpb
  mt_mbpp:ruby:bpb       mt_mbpp:rust:bpb
  mt_mbpp:scala:bpb      mt_mbpp:swift:bpb
  mt_mbpp:typescript:bpb
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Language name → HuggingFace config name in allenai/multilingual_mbpp
# Config names match the language names exactly.
LANGUAGES = [
    "bash",
    "c",
    "cpp",
    "csharp",
    "go",
    "haskell",
    "java",
    "javascript",
    "matlab",
    "php",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "swift",
    "typescript",
]


def mt_mbpp_bpb_prompt(line, task_name=None):
    """Prompt = problem description (text); gold = code solution."""
    return Doc(
        task_name=task_name,
        query=line["text"],
        choices=[line["code"]],
        gold_index=0,
    )


def _make_task(lang: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"mt_mbpp:{lang}:bpb",
        prompt_function=mt_mbpp_bpb_prompt,
        hf_repo="allenai/multilingual_mbpp",
        hf_subset=lang,
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=[Metrics.target_bits_per_byte],
        stop_sequence=None,
        version=0,
    )


TASKS_TABLE = [_make_task(lang) for lang in LANGUAGES]
