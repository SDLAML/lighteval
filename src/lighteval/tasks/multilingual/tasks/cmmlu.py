"""
name:
Cmmlu

dataset:
haonan-li/cmmlu

abstract:
CMMLU is a comprehensive Chinese evaluation benchmark covering 67 topics from
basic to advanced professional level. Named after the Chinese MMLU.

languages:
chinese

tags:
knowledge, multilingual, multiple-choice

paper:
https://arxiv.org/abs/2307.09009
"""

from string import ascii_uppercase

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import CFFormulation, MCFFormulation
from lighteval.utils.language import Language

_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]

_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]

CMMLU_SUBSETS = [
    "agronomy", "anatomy", "ancient_chinese", "arts", "astronomy",
    "business_ethics", "chinese_civil_service_exam", "chinese_driving_rule",
    "chinese_food_culture", "chinese_foreign_policy", "chinese_history",
    "chinese_literature", "chinese_teacher_qualification", "clinical_knowledge",
    "college_actuarial_science", "college_education", "college_engineering_hydrology",
    "college_law", "college_mathematics", "college_medical_statistics",
    "college_medicine", "computer_science", "computer_security", "conceptual_physics",
    "construction_project_management", "economics", "education",
    "electrical_engineering", "elementary_chinese", "elementary_commonsense",
    "elementary_information_and_technology", "elementary_mathematics", "ethnology",
    "food_science", "genetics", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_geography", "high_school_mathematics",
    "high_school_physics", "high_school_politics", "human_sexuality",
    "international_law", "journalism", "jurisprudence", "legal_and_moral_basis",
    "logical", "machine_learning", "management", "marketing", "marxist_theory",
    "modern_chinese", "nutrition", "philosophy", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_study", "sociology", "sports_science",
    "traditional_chinese_medicine", "virology", "world_history", "world_religions",
]


def _adapter(line):
    return {
        "question": line["Question"],
        "choices": [line["A"], line["B"], line["C"], line["D"]],
        "gold_idx": ascii_uppercase.index(line["Answer"]),
    }


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"cmmlu:{subset}:{suffix}",
        prompt_function=get_mcq_prompt_function(Language.CHINESE, _adapter, formulation=formulation),
        hf_repo="haonan-li/cmmlu",
        hf_subset=subset,
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=gen_size,
        metrics=metrics,
        stop_sequence=["\n"],
        version=1,
    )
    for subset in CMMLU_SUBSETS
    for suffix, formulation, metrics, gen_size in [
        ("cf",     CFFormulation(),  _CF_METRICS,              -1),
        ("mcf",    MCFFormulation(), _MCF_METRICS,             -1),
        ("mcf_em", MCFFormulation(), [Metrics.exact_match],     1),
    ]
]
