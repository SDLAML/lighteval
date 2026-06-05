"""
multilingual_aggregate.py — Aggregation helpers for multilingual task results.

Usage:
    python scripts/multilingual_aggregate.py results.json [--prefix PREFIX] [--output out.json]

    results.json   — lighteval output JSON file (the "results" field)
    --prefix       — optional task prefix filter, e.g. "global_mmlu" (default: all)
    --output       — write aggregate outputs to this JSON file (default: stdout)

How it works
------------
For standard multilingual tasks, lighteval auto-generates per-language subtask
averages such as:

    global_mmlu_fra:cf:_average|5   ← mean over all 57 MMLU subjects for French
    global_mmlu_deu:cf:_average|5   ← mean over all 57 MMLU subjects for German

This script groups those into cross-language averages such as:

    global_mmlu:cf:cross_lang_average|5

For the English-centric translation tasks, the raw task names are direction-first:

    wmt24pp:en_to_x:de_DE|0
    wmt24pp:x_to_en:de_DE|0
    wmt24pp:de_DE:_average|0

This script additionally emits language-centered aliases and translation-specific
cross-language averages such as:

    wmt24pp:de_DE:en_to_x|0
    wmt24pp:de_DE:x_to_en|0
    wmt24pp:de_DE:bidirectional_average|0
    wmt24pp:en_to_x:cross_lang_average|0
    wmt24pp:x_to_en:cross_lang_average|0
    wmt24pp:bidirectional_cross_lang_average|0
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

# All 3-letter ISO 639-3 language codes used in the Language enum.
# Derived from lighteval.utils.language.Language; kept as a static set here
# to avoid importing the full lighteval package.
_LANG_CODES = {
    "afr", "aka", "amh", "ara", "aze", "bam", "bel", "ben", "bos", "bul",
    "cat", "ces", "ckb", "cmn", "cym", "dan", "deu", "ell", "eng", "epo",
    "est", "eus", "ewe", "fas", "fin", "fra", "ful", "gla", "gle", "glg",
    "grn", "guj", "hau", "heb", "hin", "hrv", "hun", "hye", "ibo", "ind",
    "isl", "ita", "jpn", "kan", "kat", "kaz", "khm", "kin", "kir", "kon",
    "kor", "lao", "lat", "lin", "lit", "ltz", "lug", "luo", "lvs", "mal",
    "mar", "mkd", "mlt", "mri", "msa", "mya", "nep", "nld", "nor", "nob",
    "nno", "nya", "ory", "pan", "pol", "por", "pus", "ron", "run", "rus",
    "sag", "sin", "slk", "slv", "smo", "sna", "som", "sot", "spa", "sqi",
    "srp", "ssw", "swa", "swe", "tam", "tel", "tgk", "tgl", "tha", "tir",
    "ton", "tsn", "tso", "tuk", "tur", "twi", "uig", "ukr", "urd", "uzn",
    "vie", "wol", "xho", "yor", "zho", "zsm", "zul",
}

# Pattern: task names whose first component ends with _{lang_code}
# e.g. "global_mmlu_fra" → base="global_mmlu", lang="fra"
_LANG_RE = re.compile(r"^(.+?)_(" + "|".join(_LANG_CODES) + r")(:.+)?$")
_TRANSLATION_DIRECTION_RE = re.compile(r"^(?P<base>[^:]+):(?P<direction>en_to_x|x_to_en):(?P<lang>[^:]+)$")
_TRANSLATION_AVERAGE_RE = re.compile(r"^(?P<base>[^:]+):(?P<lang>[^:]+):_average$")


def _strip_lang(task_no_fewshot: str):
    """
    Given a task name without the |n suffix, return (base_name, lang_code).
    base_name is the task name with the language component replaced by nothing.
    Returns (None, None) if no language component found.

    Example:
        "global_mmlu_fra:cf:_average" → ("global_mmlu:cf:_average", "fra")
        "mlmm_arc_deu:challenge:cf:_average" → ("mlmm_arc:challenge:cf:_average", "deu")
        "gsm8k" → (None, None)
    """
    m = _LANG_RE.match(task_no_fewshot)
    if m:
        base_prefix = m.group(1)
        lang = m.group(2)
        rest = m.group(3) or ""
        return f"{base_prefix}{rest}", lang
    return None, None


def _avg(values):
    return sum(values) / len(values) if values else float("nan")


def _propagate_se(se_values):
    """SE of mean of independent estimates: sqrt(sum(SE_i^2)) / n."""
    n = len(se_values)
    return math.sqrt(sum(v ** 2 for v in se_values)) / n if n else float("nan")


def _split_fewshot(task_name: str):
    if "|" in task_name:
        task_no_fs, fs_suffix = task_name.rsplit("|", 1)
        return task_no_fs, "|" + fs_suffix
    return task_name, ""


def _is_valid_number(value):
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )


def _append_numeric_metrics(metric_lists: dict[str, list[float]], metrics: dict):
    for metric_name, metric_val in metrics.items():
        if _is_valid_number(metric_val):
            metric_lists[metric_name].append(metric_val)


def _average_metric_lists(metric_lists: dict[str, list[float]]) -> dict:
    averaged = {}
    for metric_name, vals in metric_lists.items():
        if metric_name.endswith("_stderr"):
            averaged[metric_name] = _propagate_se(vals)
        else:
            averaged[metric_name] = _avg(vals)
    return averaged


def compute_cross_lang_averages(results: dict, prefix: str | None = None) -> dict:
    """
    Given the 'results' dict from a lighteval JSON output, compute cross-language
    averages for all multilingual per-language-average tasks.

    Returns a dict mapping cross-language-average task name → metric dict.
    """
    # Only consider tasks that are per-language averages (contain :_average)
    average_tasks = {k: v for k, v in results.items() if ":_average" in k}

    if prefix:
        average_tasks = {k: v for k, v in average_tasks.items() if k.startswith(prefix)}

    # Group by (base_name_with_fewshot) where lang is stripped out
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for task_name, metrics in average_tasks.items():
        # Split off |n fewshot suffix
        if "|" in task_name:
            task_no_fs, fs_suffix = task_name.rsplit("|", 1)
            fs_part = "|" + fs_suffix
        else:
            task_no_fs = task_name
            fs_part = ""

        base_name, lang = _strip_lang(task_no_fs)
        if base_name is None:
            continue  # not a multilingual task we can aggregate

        group_key = base_name + fs_part
        for metric_name, metric_val in metrics.items():
            if isinstance(metric_val, (int, float)) and not (
                isinstance(metric_val, float) and math.isnan(metric_val)
            ):
                groups[group_key][metric_name].append(metric_val)

    # Compute cross-language averages
    cross_lang: dict[str, dict] = {}
    for group_key, metric_lists in groups.items():
        if not metric_lists:
            continue
        n_langs = max(len(v) for v in metric_lists.values())
        if n_langs < 2:
            continue  # only one language — no meaningful cross-lang average

        # Rename key: replace ":_average" with ":cross_lang_average"
        out_key = group_key.replace(":_average", ":cross_lang_average")
        cross_lang[out_key] = {}
        for metric_name, vals in metric_lists.items():
            if metric_name.endswith("_stderr"):
                cross_lang[out_key][metric_name] = _propagate_se(vals)
            else:
                cross_lang[out_key][metric_name] = _avg(vals)
        cross_lang[out_key]["_n_languages"] = n_langs

    return cross_lang


def compute_translation_aggregates(results: dict, prefix: str | None = None) -> dict:
    directional_metrics = {}
    bidirectional_averages = {}

    for task_name, metrics in results.items():
        if prefix and not task_name.startswith(prefix):
            continue

        task_no_fs, fs_part = _split_fewshot(task_name)

        directional_match = _TRANSLATION_DIRECTION_RE.match(task_no_fs)
        if directional_match:
            key = (
                directional_match.group("base"),
                directional_match.group("lang"),
                fs_part,
            )
            directional_metrics.setdefault(key, {})[
                directional_match.group("direction")
            ] = metrics
            continue

        average_match = _TRANSLATION_AVERAGE_RE.match(task_no_fs)
        if average_match:
            key = (
                average_match.group("base"),
                average_match.group("lang"),
                fs_part,
            )
            bidirectional_averages[key] = metrics

    if not directional_metrics:
        return {}

    aliases = {}
    directional_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    bidirectional_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for key, per_direction in directional_metrics.items():
        base, lang, fs_part = key

        for direction, metrics in per_direction.items():
            alias_key = f"{base}:{lang}:{direction}{fs_part}"
            aliases[alias_key] = metrics
            _append_numeric_metrics(
                directional_groups[f"{base}:{direction}:cross_lang_average{fs_part}"],
                metrics,
            )

        average_metrics = bidirectional_averages.get(key)
        if average_metrics is not None:
            aliases[f"{base}:{lang}:bidirectional_average{fs_part}"] = average_metrics
            _append_numeric_metrics(
                bidirectional_groups[f"{base}:bidirectional_cross_lang_average{fs_part}"],
                average_metrics,
            )

    for out_key, metric_lists in directional_groups.items():
        n_langs = max(len(v) for v in metric_lists.values())
        if n_langs < 2:
            continue
        aliases[out_key] = _average_metric_lists(metric_lists)
        aliases[out_key]["_n_languages"] = n_langs

    for out_key, metric_lists in bidirectional_groups.items():
        n_langs = max(len(v) for v in metric_lists.values())
        if n_langs < 2:
            continue
        aliases[out_key] = _average_metric_lists(metric_lists)
        aliases[out_key]["_n_languages"] = n_langs

    return aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_file", help="Path to lighteval results JSON file")
    parser.add_argument("--prefix", default=None, help="Filter tasks by prefix (e.g. 'global_mmlu')")
    parser.add_argument("--output", default=None, help="Write output to this JSON file (default: stdout)")
    args = parser.parse_args()

    with open(args.results_file) as f:
        data = json.load(f)

    # Support both the raw results dict and the full lighteval JSON format
    if "results" in data:
        results = data["results"]
    else:
        results = data

    aggregated = {}
    aggregated.update(compute_cross_lang_averages(results, prefix=args.prefix))
    aggregated.update(compute_translation_aggregates(results, prefix=args.prefix))

    if not aggregated:
        print("No aggregate outputs found.", file=sys.stderr)
        sys.exit(0)

    out = json.dumps(aggregated, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(out)
        print(f"Written {len(aggregated)} aggregate outputs to {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
