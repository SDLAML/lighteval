import os
os.environ["LIGHTEVAL_DISABLE_PREDICTION_CACHE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = "" # set wherever module load git points to
os.environ["VLLM_PLUGINS"] = "register_staging_moellama"

import gc
import re
import subprocess
from pathlib import Path

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.abstract_model import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

### for hybrid models:
# pip uninstall -y mamba-ssm selective-scan causal-conv1d
# pip install --no-cache-dir --no-binary :all: --no-build-isolation "mamba-ssm[causal-conv1d]"

### also don't forget before running:
# module load git 

######## GENERAL CONFIGURATION  ########

SUBDIR_PREFIX = "test-"
ENFORCE_EAGER_MODELS = ['Trinity']
HF_BACKEND_MODELS = ['granite', 'Ministral', 'Falcon', 'Apertus']
BATCH_SIZE = 32 # only used for HF backend
DP_SIZE = 4 # only used for vLLM backend

######## EVALUATION CONFIGURATION  ########

OVERRIDE_CHAT_TEMPLATE = False # False for base, don't forget to change for instruction-tuned models!
SEED = 1234
TEMPERATURE = 0
TOP_P = None
MAX_MODEL_LENGTH = 4096
MAX_NEW_TOKENS = None
MAX_SAMPLES = None
SKIP_SPECIAL_TOKENS = True # vllm default
SPACES_BETWEEN_SPECIAL_TOKENS = True # vllm default

# OVERRIDE_CHAT_TEMPLATE = True # False for base, don't forget to change for instruction-tuned models!
# SEED = 1234
# TEMPERATURE = 0.6
# TOP_P = 0.95
# MAX_MODEL_LENGTH = 65536 # 65536 // 32768
# MAX_NEW_TOKENS = 32768 # 32768 // 16384
# MAX_SAMPLES = None
# SKIP_SPECIAL_TOKENS = False
# SPACES_BETWEEN_SPECIAL_TOKENS = False


######## MODEL NAMES  ########

MODEL_NAMES = [
                "allenai/OLMoE-1B-7B-0125", 
                "HuggingFaceTB/SmolLM3-3B-Base",
                ##
                "Qwen/Qwen3-8B-Base",
                "XiaomiMiMo/MiMo-7B-Base",
                "swiss-ai/Apertus-8B-2509",
                "marin-community/marin-8b-base", 
                "EssentialAI/rnj-1",
                "utter-project/EuroLLM-9B", 
            ]

TASKS = [
            # "hellaswag|5", "siqa|5", "commonsenseqa|5", "openbookqa|5", 
            # "squad_v2|5", "drop|5", 
            # "triviaqa|5", "wikifact|5", "truthfulqa:mc|0", "popqa|5",
            # "mmlu|5", "sciq:mc|5",
            # "arc:easy:mcf|5", "arc:challenge:mcf|5", 
            # "gsm8k|0", "gsm_plus|0", "math_500|0", "math|0",
            # "agieval_eng_em|0",
]

# TASKS += [
#             f"global_mmlu_all_{lang}_mcf|5" for lang in [
#                "ces", "deu", "fra", "pol", "spa", "swe",
#                "ara", "hin", "rus", "tur", "vie", "zho",
#             #    "srp",
#             ]
#         ] 

# TASKS += [
#             f"mlmm_hellaswag_{lang}_cf|0" for lang in [
#                 "dan", "deu", "fra", "hrv", "hun", "spa", "isl",
#                 "ara", "hin", "rus", "vie", "zho",
#             ]
#         ] 

# TASKS = [
#           "bbq|0", "toxigen|0", "bold|0", "civil_comments|0", "real_toxicity_prompts|0", "ethics|0",
#          ]

def _get_git_commit_short() -> str:
    """Get the first 7 characters of the current git commit hash."""
    try:
        repo_dir = Path(__file__).parent.parent  # lighteval repo root
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:7]
    except Exception:
        return "unknown"

def _safe_name(s: str, max_len: int = 180) -> str:
    """Filesystem-safe name for model paths / task strings (handles / : | etc.)."""
    s = s.strip().strip("/")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:max_len]

def _dist_info():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size(), dist
    except Exception:
        pass
    return 0, 1, None

def eval_one(model_name: str, tasks: str):
    rank, world, dist = _dist_info()

    backend = "hf" if any(n in model_name for n in HF_BACKEND_MODELS) else "vllm"
    if backend == "hf":
        assert not ',' in tasks, "comma-separated tasks are broken for the HF backend, please run them one at a time"

    #### Configure pipeline and model ####

    pipeline_params = PipelineParameters(
        launcher_type= ParallelismManager.VLLM if backend == "vllm" else ParallelismManager.ACCELERATE,
        load_tasks_multilingual=True,
        max_samples=MAX_SAMPLES,
    )

    model_cfg_kwargs = dict(
        model_name=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        override_chat_template=OVERRIDE_CHAT_TEMPLATE,
        generation_parameters=GenerationParameters(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
        skip_special_tokens=SKIP_SPECIAL_TOKENS,
        spaces_between_special_tokens=SPACES_BETWEEN_SPECIAL_TOKENS,
        ),
    )
    
    if backend == "vllm":
        model_cfg = VLLMModelConfig(
            **model_cfg_kwargs,
            max_model_length=MAX_MODEL_LENGTH,
            seed=SEED,
            enforce_eager=True if any(x in model_name for x in ENFORCE_EAGER_MODELS) else False,
            distributed_backend="mp",
            data_parallel_size=DP_SIZE,
            # enable_prefix_caching=False,
        )
    elif backend == "hf":
        model_cfg = TransformersModelConfig(
            **model_cfg_kwargs,
            batch_size=BATCH_SIZE,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    #### Loop over tasks ####

    if isinstance(tasks, list) and backend == "vllm":
        tasks = ','.join(tasks)
    
    for task in tasks if isinstance(tasks, list) else [tasks]:
        
        if rank == 0:
            print(f"\n{'='*100}")
            print(f"Evaluating model: {model_name}")
            print(f"Task: {task}")
            print(f"World size: {world}")
            print(f"{'='*100}\n")
        
        out_dir = Path("results") / _safe_name(model_name) / task / f'{SUBDIR_PREFIX}{_get_git_commit_short()}'
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_tracker = EvaluationTracker(
            output_dir=str(out_dir), 
            save_details=True, 
            results_path_template="{output_dir}/results",
        )

        pipeline = Pipeline(
            tasks=task,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=eval_tracker,
            model_config=model_cfg,
        )
        pipeline.evaluate()
        pipeline.save_and_push_results()

        if rank == 0:
            pipeline.show_results()

        # Make sure all ranks finish this task before moving on
        if dist is not None:
            dist.barrier()

        # Try hard to free memory between runs
        try:
            import torch

            del pipeline
            del eval_tracker
            torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()

        if dist is not None:
            dist.barrier()


def main():
    for model_name in MODEL_NAMES:
        eval_one(model_name, TASKS)

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass