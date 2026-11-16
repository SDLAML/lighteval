# MIT License
#
# titan_eval (owner 11-16): the answer-budget rule of the long-context chat tasks with `over_context: zero` (FRAMES, MRCR chat, RULER chat, HELMET chat).
#
# OLMES `oe_eval/tasks/oe_eval_tasks/openai_mrcr.py:41-46` lets the answer budget shrink to what the context leaves (`"max_gen_toks": 16384`, `"min_acceptable_gen_toks": 1024`, "Truncate
# instances which cause max_gen_toks to be <1024"). Ours: budget = min(task budget, context - prompt - margin); the item is NOT generated and scores 0 only if that is below the floor
# (1,024 for MRCR / FRAMES, the subtask's / dataset's own answer size for RULER / HELMET); an item whose budget is below the task budget but at least the floor is generated with the
# smaller budget (`specific["max_new_tokens"]`, the vLLM model gives that request its own cap) and counted as reduced.
"""The pure rule (no tokenizer, no environment): `fit_answer`."""

import os


ENV_MAX_MODEL_LEN = "TITAN_EVAL_MAX_MODEL_LEN"
SKIP_USER = "Reply with the word OK."      # the prompt of an item that is not generated: never sent (specific["no_generation"]), kept tiny for the details files
MARGIN = 64                      # tokens kept free for the chat template and the re-tokenisation (owner 11-16)
CHAT_TEMPLATE_TOKENS = 16        # an upper bound of what a chat template adds around ONE user message (Qwen3.5 10 / 12, hamburger_pt 10 / 12), for the estimates of RULER / HELMET


def run_context() -> int | None:
    """The run's window (vllm.max_model_len exported by the titan_eval adapter), None when the run did not export it (unit tests, `lighteval tasks inspect`)."""
    v = os.environ.get(ENV_MAX_MODEL_LEN)
    return int(v) if v else None


def no_fit(task_budget: int) -> dict:
    """Nothing known about the run (no tokenizer / context): the whole budget, not over, not reduced."""
    return {"answer_budget": task_budget, "task_budget": task_budget, "over_context": False, "reduced_budget": False}


def fit_answer(n_prompt: int, task_budget: int, floor: int, margin: int = MARGIN, ctx: int | None = None) -> dict:
    """The OLMES-style rule. -> {answer_budget (what the request gets; 0 when over), task_budget, over_context, reduced_budget}.
    Without a context (None and no run variable) nothing is marked: the whole budget, not over."""
    ctx = run_context() if ctx is None else ctx
    if ctx is None:
        return no_fit(task_budget)
    budget = min(task_budget, ctx - n_prompt - margin)
    over = budget < floor
    return {"answer_budget": 0 if over else budget, "task_budget": task_budget, "over_context": over, "reduced_budget": (not over) and budget < task_budget}


def mark(spec: dict, fit: dict) -> dict:
    """Write the result of `fit_answer` into a doc's `specific`: `over_context`, `reduced_budget`, `answer_budget`, `task_budget`; `no_generation` for an item at zero (no request);
    `max_new_tokens` for a reduced item (the request's own cap, see vllm_model._greedy_until)."""
    spec.update(over_context=fit["over_context"], reduced_budget=fit["reduced_budget"], answer_budget=fit["answer_budget"], task_budget=fit["task_budget"])
    if fit["over_context"]:
        spec["no_generation"] = True
    elif fit["reduced_budget"]:
        spec["max_new_tokens"] = fit["answer_budget"]
    return spec
