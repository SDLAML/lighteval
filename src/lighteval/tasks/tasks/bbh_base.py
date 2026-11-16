"""
name:
BIG-Bench Hard, base-model protocol (3-shot chain of thought with the official exemplars)

dataset:
lukaemon/bbh

abstract:
BIG-Bench Hard (Suzgun et al. 2022) as the paper evaluates base models: 27 subtasks, each prompted with its OFFICIAL chain-of-thought prompt
(`cot-prompts/<subtask>.txt` of github.com/suzgunmirac/BIG-Bench-Hard: the one-line task description and 3 hand-written exemplars, vendored in
bbh_cot_demos.py), then the question as "Q: <input>\\nA: Let's think step by step." The answer is what follows "the answer is" in the generation
(the exemplars all end "So the answer is X."): the rest of that line without one trailing period, compared with the target by EXACT match (case and
spelling as the dataset writes the target: "(B)", "True", "yes", ...). Greedy decoding, at most 1024 new tokens (lm-evaluation-harness `bbh_cot_fewshot`),
stopped at a blank line or a new "Q:" line (the stops of lm-evaluation-harness and OLMES, see STOP_SEQUENCES). The score is the macro mean over the 27 subtasks (6,511 questions; 250 each except causal_judgement 187,
penguins_in_a_table 146, snarks 178). Metrics: em, answer_found (share of outputs with a non-empty "the answer is" line: 1 - answer_found is the format-fail rate).
Not the base task `bigbench_hard:*` of this fork (random exemplars from the test split, no official CoT text, stop sequence "Q" that cuts at any capital Q; since the opt_moe merge of 11-16 it holds multiple-choice formats only).

languages:
english

tags:
reasoning

paper:
https://arxiv.org/abs/2210.09261
"""

import re
from functools import partial
from typing import Any

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.bbh_cot_demos import BBH_COT_DEMOS
from lighteval.tasks.tasks.bigbench_hard import _BBH_SUBSETS


REPO = "lukaemon/bbh"
REVISION = "982bb89fd79532a8ac676a61fc42eb1aeec63f99"  # the pin of bbh_cot_chat too (test split, 27 subsets)
# Generation settings, from the two harnesses that run this protocol (owner 11-16: every setting with its source):
#   lm-evaluation-harness `lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml` @ d6de8164, lines 12-17:  max_gen_toks: 1024, until: ["</s>", "Q", "\n\n"], do_sample: false, temperature: 0.0
#   OLMES `oe_eval/tasks/oe_eval_tasks/bbh.py` @ 5a51f502, lines 88-98:  num_shots 3, use_cot, max_gen_toks 512, temperature 0.0, stop_sequences ["\n\n"]
# Taken: 1024 new tokens (lm-evaluation-harness; OLMES's 512 is the lower of the two) and the stops ["\n\n", "\nQ:"]: both harnesses stop at a blank line (the exemplars never contain one: the
# answer line is the last line of the block); "</s>" is the end-of-text token vLLM stops at anyway; the bare "Q" of lm-evaluation-harness is DROPPED because it stops at ANY capital Q in the
# reasoning ("Q1", "QUESTION", names); "\nQ:" is kept for a model that writes the next question without a blank line before it.
MAX_NEW_TOKENS = 1024
STOP_SEQUENCES = ["\n\n", "\nQ:"]
TASK_VERSION = 0
_STEP = "Let's think step by step."


def _demo_rows(subset: str) -> list[dict]:
    """The 3 official exemplars of a subtask in the file's order: rows with the fields `input` and `answer` (the text after "A: ")."""
    return [dict(shot) for shot in BBH_COT_DEMOS[subset]["shots"]]


def bbh_base_prompt(line, task_name: str = None, *, subset: str):
    demo = BBH_COT_DEMOS[subset]
    if line.get("__few_shots"):
        # "<Q: ...>\nA:" + " " + "Let's think step by step.<newline or space><reasoning>. So the answer is X." (lighteval joins the query and the gold with one space)
        return Doc(task_name=task_name, query=f"Q: {line['input']}\nA:", choices=[line["answer"]], gold_index=0)
    # The question ends the way the exemplars' "A:" line does: a newline after "Let's think step by step." (26 subtasks) or nothing (sports_understanding, whose
    # exemplars continue on the same line: no trailing blank, the model writes it).
    tail = "\n" if demo["shots"][0]["answer"][len(_STEP)] == "\n" else ""
    return Doc(
        task_name=task_name,
        query=f"Q: {line['input']}\nA: {_STEP}{tail}",
        choices=[line["target"].strip()],
        gold_index=0,
        instruction=demo["description"],
        specific={"subset": subset},
    )


_ANSWER = re.compile(r"the answer is[ \t]*([^\n]*)")


def extract_answer(text: str) -> str | None:
    """What follows the FIRST "the answer is" of the generation, up to the end of that line, without ONE trailing period; None when there is none or it is empty."""
    match = _ANSWER.search(text or "")
    if match is None:
        return None
    answer = match.group(1).strip()
    if answer.endswith("."):
        answer = answer[:-1].rstrip()
    return answer or None


class CotExactMatch(SampleLevelComputation):
    def compute(self, doc: Doc, model_response, **kwargs) -> dict[str, Any]:
        text = (model_response.final_text[0] if model_response.final_text else "") or ""
        answer = extract_answer(text)
        return {"em": float(answer is not None and answer == doc.choices[doc.gold_index]), "answer_found": float(answer is not None)}


BBH_BASE_METRIC = SampleLevelMetricGrouping(
    metric_name=["em", "answer_found"],
    sample_level_fn=CotExactMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={"em": np.mean, "answer_found": np.mean},
    higher_is_better={"em": True, "answer_found": True},
)

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"bbh_base:{subset}:cot",
        prompt_function=partial(bbh_base_prompt, subset=subset),
        hf_repo=REPO,
        hf_subset=subset,
        hf_revision=REVISION,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_rows=partial(_demo_rows, subset),
        few_shots_select="sequential",  # the first K (3) official exemplars, in the file's order
        generation_size=MAX_NEW_TOKENS,
        metrics=[BBH_BASE_METRIC],
        stop_sequence=STOP_SEQUENCES,
        version=TASK_VERSION,
    )
    for subset in _BBH_SUBSETS
]
