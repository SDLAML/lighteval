"""
name:
Mmlu Redux

dataset:
edinburgh-dawg/mmlu-redux-2.0

abstract:
MMLU-Redux is a subset of 5,700 manually re-annotated questions across 57 MMLU subjects.
The corrected tasks (mcf_em, cf, mcf) are defined in mmlu.py under the mmlu_redux: prefix.

languages:
english

tags:
general-knowledge, knowledge, multiple-choice

paper:
https://arxiv.org/abs/2406.04127
"""

# All mmlu_redux tasks (mcf_em, cf, mcf) are defined in mmlu.py
# under the mmlu_redux:{subset}:{variant} naming scheme.
# This file is intentionally empty to avoid the legacy mmlu_redux_2 tasks
# which used incompatible metric/generation_size combinations.

TASKS_TABLE = []
