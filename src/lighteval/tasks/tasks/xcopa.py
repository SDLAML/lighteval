"""
name:
Xcopa

dataset:
cambridgeltl/xcopa

abstract:
XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning The Cross-lingual
Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability
of machine learning models to transfer commonsense reasoning across languages.

languages:
english

tags:
commonsense, multilingual, multiple-choice, reasoning

paper:
https://arxiv.org/abs/2005.00333
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


### from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/xcopa/utils.py
CONNECTORS = {
    "en": {
        "cause": "because",
        "effect": "therefore",
    },

    "et": {
        "cause": "sest",
        "effect": "seetõttu",
    },

    "ht": {
        "cause": "poukisa",
        "effect": "donk sa",
    },

    "it": {
        "cause": "perché",
        "effect": "quindi",
    },

    "id": {
        "cause": "karena",
        "effect": "maka",
    },

    "qu": {
        "cause": "imataq",
        "effect": "chaymi",
    },

    "sw": {
        "cause": "kwa sababu",
        "effect": "kwa hiyo",
    },

    "zh": {
        "cause": "因为",
        "effect": "所以",
    },

    "ta": {
        "cause": "காரணமாக",
        "effect": "எனவே",
    },

    "th": {
        "cause": "เพราะ",
        "effect": "ดังนั้น",
    },

    "tr": {
        "cause": "çünkü",
        "effect": "bu yüzden",
    },

    "vi": {
        "cause": "bởi vì",
        "effect": "vì vậy",
    },
}

def _lower_first_char(text):
    """Lowercase the first character of text for natural sentence continuation."""
    if not text:
        return text
    return text[0].lower() + text[1:]


def _turkish_lower_first_char(text):
    """Turkish-specific first character lowercasing (İ→i, I→ı)."""
    if not text:
        return text
    first = text[0]
    if first == "İ":
        return "i" + text[1:]
    elif first == "I":
        return "ı" + text[1:]
    return first.lower() + text[1:]


def xcopa_prompt(line, task_name: str = None, lang: str = "en"):
    """Build a prompt where premise + connector form the query and choices
    are preprocessed for natural sentence continuation.

    Language-specific handling:
    - zh: Chinese comma ，, no spaces between tokens, no case change
    - th: space-separated clauses, no comma, no case change
    - ta: no case distinction, comma-separated
    - tr: Turkish-specific İ/I lowercasing rules
    - others (en, et, ht, it, id, qu, sw, vi): standard Latin lowercase
    """
    premise = line["premise"].strip()
    connector = CONNECTORS[lang][line["question"]]
    choice1 = line["choice1"]
    choice2 = line["choice2"]

    if lang == "zh":
        # Chinese: use Chinese comma ，, no spaces, no case change
        query = f"{premise.rstrip('。')}，{connector}"
        choices = [choice1, choice2]
    elif lang == "th":
        # Thai: no case distinction, space-separated clauses, no comma
        query = f"{premise.rstrip('.')} {connector}"
        choices = [f" {choice1}", f" {choice2}"]
    elif lang == "ta":
        # Tamil: no case distinction, comma-separated
        query = f"{premise.rstrip('.')}, {connector}"
        choices = [f" {choice1}", f" {choice2}"]
    elif lang == "tr":
        # Turkish: special İ/I lowercase rules
        query = f"{premise.rstrip('.')}, {connector}"
        choices = [f" {_turkish_lower_first_char(choice1)}", f" {_turkish_lower_first_char(choice2)}"]
    else:
        # Latin-script languages with standard case: en, et, ht, it, id, qu, sw, vi
        query = f"{premise.rstrip('.')}, {connector}"
        choices = [f" {_lower_first_char(choice1)}", f" {_lower_first_char(choice2)}"]

    # gold_index = int(line["label"]) - 1 if isinstance(line["label"], str) else int(line["label"])
    gold_index = int(line["label"])
    return Doc(task_name=task_name, query=query, choices=choices, gold_index=gold_index)


TASKS_TABLE = [
    LightevalTaskConfig(
        name="xcopa:en",
        prompt_function=lambda line, task_name: xcopa_prompt(line, task_name, lang="en"),
        hf_repo="cambridgeltl/xcopa",
        hf_subset="translation-it",
        hf_avail_splits=["test", "train", "validation"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        metrics=[Metrics.loglikelihood_acc],
        stop_sequence=["\n"],
        version=0,
    )
]

TASKS_TABLE += [
    LightevalTaskConfig(
        name=f"xcopa:{lang}",
        prompt_function=lambda line, task_name: xcopa_prompt(line, task_name, lang=lang),
        hf_repo="cambridgeltl/xcopa",
        hf_subset=lang,
        hf_avail_splits=["test", "train", "validation"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=-1,
        metrics=[Metrics.loglikelihood_acc],
        stop_sequence=["\n"],
        version=0,
    )
    for lang in ["et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]
]