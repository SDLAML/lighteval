"""
name:
GeoBench Geoscience QA

dataset:
titaneval_local (local parquet - data/titaneval/geobench.parquet)

abstract:
Geoscience / geography multiple-choice benchmark: the AP-Test objective questions of the GeoBenchmark of the K2 geoscience LLM
paper (Deng et al. 2023; HF daven3/geobench @ fa6f583a, file geobenchmark_apstudy.json, Apache-2.0 on the card), 5 options each.
This is NOT GeoBench-VLM (the image benchmark): the TitanEval-MCQ collector names the wrong source, the content is the text
GeoBenchmark. The local file has 1,392 rows = the 1,395 upstream rows without the 3 that have an empty question stem, in the
upstream order (checked row by row).

Version 1 (owner 2026-11-13): rows that cannot be answered or are malformed are dropped (`drop_reason`, the rules are below):
155 of 1,392 dropped (placeholder 21, table_figure 25, exhibit 103, empty_option 3, duplicate_option 3), 1,237 scored (also in
configs/tasks/lighteval/geobench_mcf_em.yaml, notes). The same filtered split is the few-shot pool, so a broken item is never a
demonstration either. The dropped rows are listed (index in the file, reason, question start) in
docs/evals/tasks/_raw/geobench_mcf_em-dropped.tsv in titan_eval.

Dropped rows, in this order of checking:
  placeholder (21)       the literal token "asdf" in the question (an unfilled "see the table below asdf" marker);
  table_figure (25)      the question refers to a table / figure / graph / chart that is not in the text ("the table below",
                         "the graph shown", "Figure 2", "the chart above");
  exhibit (103)          the same for a map / image / photo / picture / diagram / graphic / cartoon / population pyramid, for an item
                         labelled "Source A", "area labeled", "depicted / pictured / portrayed", "of this type", "the document above",
                         "this urban model", "these religious structures";
  empty_option (3)       one option is empty (the options of the item were lost);
  duplicate_option (3)   two options have the same text after strip + casefold (options cut at a full stop such as "U.S." fall here).
Not dropped: a generic word ("water table", "figure out", "Which of the following maps would have the largest scale?").

Few-shot pool = the scored rows (`random_sampling`, the evaluated item is excluded); 5 shots in the suite.

languages:
english

tags:
geoscience, multiple-choice, qa, science
"""

import re

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.domain_drops import KeepRows

_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]
_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def geobench_cf_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in line["choices"]],
        gold_index=line["answer_index"],
    )


def geobench_mcf_prompt(line, task_name: str = None):
    choices = line["choices"]
    labels = list("ABCDEFGHIJ"[: len(choices)])
    options = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=line["answer_index"],
    )


# ---- the drop rule (a text rule on the question; the file has no field that marks such rows) ---------------------------------------
_TABLE = r"(?:tables?|figures?|graphs?|charts?)"
_EXHIBIT = r"(?:diagrams?|images?|photos?|photographs?|pictures?|maps?|graphics?|cartoons?|illustrations?|pyramids?)"
# a determiner that points at a given exhibit ("the", "this", "these", "the following", "above", "provided", ...);
# "of the following maps" is a list of options, not an exhibit
_POINT = r"(?:the|this|these|those|(?<!of the )following|above|below|first|second|third|fourth|both|each|provided|given)"


def _pointing(noun: str) -> "re.Pattern":
    """"the table", "the bar graph", "the second image", "the table below", "graph provided", "above map"."""
    return re.compile(
        rf"\b{_POINT}\s+(?:(?!following\b)\w+\s+)?{noun}\b|\b{noun}\s+(?:above|below|provided|shown|given)\b|\b(?:above|below)\s+{noun}\b",
        re.I,
    )


_PLACEHOLDER = re.compile(r"\basdf\b", re.I)
_TABLE_FIGURE = [_pointing(_TABLE), re.compile(r"\b(?:figure|table|fig\.)\s*\d", re.I)]
_OTHER_EXHIBIT = [
    _pointing(_EXHIBIT),
    re.compile(r"\b(?:pictured|depicted|portrayed)\b", re.I),
    re.compile(
        r"\blabel(?:l)?ed\b|\bSources?\s+[A-D]\b|\bcircled\b|\bDiagram [A-E]\b|\b(?:such|these|those|this) (?:maps?|graphs?|images?)\b|\bof this type\b",
        re.I,
    ),
    re.compile(
        r"\b(?:model|structure|building)\s+(?:above|below)\b|\b(?:shown|seen|illustrated|displayed|represented)\s+(?:above|below)\b"
        r"|\b(?:passage|excerpt|document|quotation|quote)s?\s+(?:above|below)\b|\b(?:in|of|from) the (?:passage|excerpt|document)\b"
        r"|\bbar graphs?\b|\bWhich (?:diagram|graph|map|image|photo)\b|^The above\b|^This model\b"
        r"|\b(?:this|these)\s+(?:\w+\s+){0,2}(?:structures?|urban models?)\b",
        re.I,
    ),
]


def drop_reason(row):
    """None = scored, else the reason code (see the module docstring)."""
    question = row["question"] or ""
    options = [str(c) for c in row["choices"]]
    if _PLACEHOLDER.search(question):
        return "placeholder"
    if any(p.search(question) for p in _TABLE_FIGURE):
        return "table_figure"
    if any(p.search(question) for p in _OTHER_EXHIBIT):
        return "exhibit"
    if any(not o.strip() for o in options):
        return "empty_option"
    if len({o.strip().casefold() for o in options}) < len(options):
        return "duplicate_option"
    return None


# The digest of the patterns is part of the datasets cache key (see domain_drops); bump `version` and `rule` with every change of the rule.
_KEEP = KeepRows(
    "geobench-v1: placeholder, table/figure/exhibit reference, empty or duplicate option",
    drop_reason,
    parts=[_PLACEHOLDER, *_TABLE_FIGURE, *_OTHER_EXHIBIT],
)


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"geobench:{suffix}",
        prompt_function=fn,
        hf_repo="titaneval_local",
        hf_subset="geobench",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split="test",
        few_shots_select="random_sampling",
        generation_size=gen,
        metrics=metrics,
        hf_filter=_KEEP,
        stop_sequence=["\n"],
        version=1,
    )
    for suffix, fn, metrics, gen in [
        ("cf",     geobench_cf_prompt,  _CF_METRICS,          -1),
        ("mcf",    geobench_mcf_prompt, _MCF_METRICS,         -1),
        ("mcf_em", geobench_mcf_prompt, [Metrics.exact_match],  1),
    ]
]
