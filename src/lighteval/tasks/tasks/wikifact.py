"""
name:
Wikifact

dataset:
lighteval/wikifact

abstract:
Extensively test factual knowledge.

languages:
english

tags:
factuality, knowledge

paper:
https://aclanthology.org/D19-1250/
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


SUBSETS = [
    "applies_to_jurisdiction",
    "author",
    "award_received",
    "basic_form_of_government",
    "capital",
    "capital_of",
    "central_bank",
    "composer",
    "continent",
    "country",
    "country_of_citizenship",
    "country_of_origin",
    "creator",
    "currency",
    "defendant",
    "developer",
    "diplomatic_relation",
    "director",
    "discoverer_or_inventor",
    "drug_or_therapy_used_for_treatment",
    "educated_at",
    "employer",
    "field_of_work",
    "genetic_association",
    "genre",
    "has_part",
    "head_of_government",
    "head_of_state",
    "headquarters_location",
    "industry",
    "influenced_by",
    "instance_of",
    "instrument",
    "language_of_work_or_name",
    "languages_spoken_written_or_signed",
    "laws_applied",
    "located_in_the_administrative_territorial_entity",
    "location",
    "location_of_discovery",
    "location_of_formation",
    "majority_opinion_by",
    "manufacturer",
    "measured_physical_quantity",
    "medical_condition_treated",
    "member_of",
    "member_of_political_party",
    "member_of_sports_team",
    "movement",
    "named_after",
    "native_language",
    "occupation",
    "office_held_by_head_of_government",
    "office_held_by_head_of_state",
    "official_language",
    "operating_system",
    "original_language_of_film_or_TV_show",
    "original_network",
    "overrules",
    "owned_by",
    "part_of",
    "participating_team",
    "place_of_birth",
    "place_of_death",
    "plaintiff",
    "position_held",
    "position_played_on_team",
    "programming_language",
    "recommended_unit_of_measurement",
    "record_label",
    "religion",
    "repealed_by",
    "shares_border_with",
    "solved_by",
    "statement_describes",
    "stock_exchange",
    "subclass_of",
    "subsidiary",
    "symptoms_and_signs",
    "therapeutic_area",
    "twinned_administrative_body",
    "work_location",
]


def wikifact_gen_prompt(line, task_name: str = None):
    """GenQA variant: generate the fact, score F1/EM against all references."""
    refs = line["references"]
    # query already ends with a space, so references need no extra leading space
    return Doc(
        task_name=task_name,
        query=f"{line['question']} ",
        choices=list(refs),
        gold_index=list(range(len(refs))),
    )


def wikifact_bpb_prompt(line, task_name: str = None):
    """BPB variant: score the first reference as the gold continuation."""
    refs = line["references"]
    if not refs:
        return None
    return Doc(
        task_name=task_name,
        query=f"{line['question']} ",
        choices=[refs[0]],
        gold_index=0,
    )


def _gen_config(subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"wikifact:{subset}:gen",
        prompt_function=wikifact_gen_prompt,
        hf_repo="lighteval/wikifact",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        # test-only dataset; two relations have just 5 rows. "from_train" sampling draws
        # num_fewshot (not num_fewshot+1), so 5-shot doesn't overflow those tiny pools.
        few_shots_select="random_sampling_from_train",
        generation_size=8,
        metrics=[Metrics.qa_f1, Metrics.qa_em],
        stop_sequence=["\n"],
        version=1,
    )


def _bpb_config(subset: str) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"wikifact:{subset}:bpb",
        prompt_function=wikifact_bpb_prompt,
        hf_repo="lighteval/wikifact",
        hf_subset=subset,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        # test-only dataset; two relations have just 5 rows. "from_train" sampling draws
        # num_fewshot (not num_fewshot+1), so 5-shot doesn't overflow those tiny pools.
        few_shots_select="random_sampling_from_train",
        generation_size=-1,
        metrics=[Metrics.target_bits_per_byte],
        stop_sequence=["\n"],
        version=1,
    )


TASKS_TABLE = [config for subset in SUBSETS for config in (_gen_config(subset), _bpb_config(subset))]
