"""
MT-MBPP (MultiPL-E) BPB evaluation tasks for lighteval.

Dataset : nuprl/MultiPL-E (HuggingFace), one subset per language
ICL     : 3 shots sampled from the test split (no dedicated train split)
Metric  : BPB = -log2 p(solution | prompt) / bytes(solution)

The prompt field in MultiPL-E already contains the language-appropriate
function signature / docstring.  The solution field contains the reference
implementation body that follows the prompt (NOT the full file).

17 language subtasks:
  multipl_e:mbpp:bash:bpb       multipl_e:mbpp:c:bpb
  multipl_e:mbpp:cpp:bpb        multipl_e:mbpp:csharp:bpb
  multipl_e:mbpp:go:bpb         multipl_e:mbpp:haskell:bpb
  multipl_e:mbpp:java:bpb       multipl_e:mbpp:javascript:bpb
  multipl_e:mbpp:matlab:bpb     multipl_e:mbpp:php:bpb
  multipl_e:mbpp:python:bpb     multipl_e:mbpp:r:bpb
  multipl_e:mbpp:ruby:bpb       multipl_e:mbpp:rust:bpb
  multipl_e:mbpp:scala:bpb      multipl_e:mbpp:swift:bpb
  multipl_e:mbpp:typescript:bpb

Note: availability of each language subset depends on the nuprl/MultiPL-E
dataset.  Bash, C, and MATLAB may not be present; if the HF subset is
missing lighteval will raise a dataset-not-found error for that task.
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Human-readable language name → HuggingFace subset name in nuprl/MultiPL-E
# Format: "mbpp-{lang_code}"
LANGUAGE_SUBSETS: dict[str, str] = {
    "bash":       "mbpp-bash",
    "c":          "mbpp-c",
    "cpp":        "mbpp-cpp",
    "csharp":     "mbpp-cs",
    "go":         "mbpp-go",
    "haskell":    "mbpp-hs",
    "java":       "mbpp-java",
    "javascript": "mbpp-js",
    "matlab":     "mbpp-matlab",
    "php":        "mbpp-php",
    "python":     "mbpp-py",
    "r":          "mbpp-r",
    "ruby":       "mbpp-rb",
    "rust":       "mbpp-rs",
    "scala":      "mbpp-scala",
    "swift":      "mbpp-swift",
    "typescript": "mbpp-ts",
}


def multipl_e_bpb_prompt(line, task_name=None):
    """Prompt = language-specific function signature/docstring; gold = solution body."""
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        # `solution` in MultiPL-E contains only the implementation body
        # (the part that follows `prompt`), not the full file.
        choices=[line["solution"]],
        gold_index=0,
    )


def _make_task(lang_name: str, hf_subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"multipl_e:mbpp:{lang_name}:bpb",
        prompt_function=multipl_e_bpb_prompt,
        hf_repo="nuprl/MultiPL-E",
        hf_subset=hf_subset,
        # MultiPL-E MBPP typically ships as a single "test" split.
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=-1,
        metrics=[Metrics.target_bits_per_byte],
        stop_sequence=None,
        version=0,
    )


TASKS_TABLE = [
    _make_task(lang_name, hf_subset)
    for lang_name, hf_subset in LANGUAGE_SUBSETS.items()
]
