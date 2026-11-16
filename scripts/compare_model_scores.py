#!/usr/bin/env python3
r"""Compare saved model evaluations by category in CSV and Markdown.

Run from the repository root:
    python scripts/compare_model_scores.py model-a model-b \
        --results-dir ~/results -o reports/comparison.csv

Writes reports/comparison.csv and reports/comparison.md. --markdown-output can
choose a separate Markdown path. Both reports use 0-1 scores rounded to two
decimal places by default; --percentages switches to 0-100 scores. --raw
explicitly selects the default 0-1 format. Standard deviations use the same
scale as scores and are also rounded to two decimal places.

Models may be exact names, original model paths, or Hugging Face IDs, following
scripts/eval.py's result-directory naming convention. Model weights need not
exist locally. Saved results_*.json files are searched recursively per model.
Select the few-shot suffix with --fewshot (default: 5). Repeated task results
are errors unless --duplicates latest selects by the saved filename timestamp.

Edit CATEGORIES below to change task membership, display labels, or metrics.
The defaults match compare_models.py: 15 General QA and 13 Scientific QA tasks.
ARC mean and SuperGPQA discipline mean are excluded to avoid counting them
alongside their component benchmarks. Benchmark means are read directly from
JSON; category aggregates are equal-weight macro means of the listed tasks.
Missing scores and category aggregates with missing tasks are blank. Standard
deviations are blank if any requested model score is missing. Each category
also includes the mean of per-task model ranks (1 is best; ties share their
average rank). Ranks are never percentage-scaled and are blank if any model
is missing a task in the category. --strict fails before writing either
report if any selected score is missing.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


# Each entry is (display label, exact saved task ID without |N, metric key).
CATEGORIES = {
    "General QA": (
        ("HellaSwag", "hellaswag:mcf_em", "em"),
        ("SIQA", "siqa:mcf_em", "em"),
        ("CommonsenseQA", "commonsenseqa:mcf_em", "em"),
        ("OpenBookQA", "openbookqa:mcf_em", "em"),
        ("WinoGrande", "winogrande:mcf_em", "em"),
        ("PIQA", "piqa:mcf_em", "em"),
        ("Jeopardy MC", "jeopardy_mc:mcf", "acc"),
        ("SQuAD v2 (answerable only)", "squad_v2:gen", "f1"),
        ("DROP", "drop:gen", "f1"),
        ("TriviaQA", "triviaqa:gen", "f1"),
        ("WikiFact (mean)", "wikifact:gen:_average", "f1"),
        ("TruthfulQA (generative)", "truthfulqa:gen", "f1"),
        ("PopQA", "popqa:gen", "f1"),
        ("Natural Questions", "natural_questions:gen", "f1"),
        ("CoQA", "coqa:gen", "f1"),
    ),
    "Scientific QA": (
        ("MMLU (mean)", "mmlu:mcf_em:_average", "em"),
        ("SciQ", "sciq:mcf_em", "em"),
        ("ARC Easy", "arc:easy:mcf_em", "em"),
        ("ARC Challenge", "arc:challenge:mcf_em", "em"),
        ("AGIEval English (mean)", "agieval_eng:mcf_em:_average", "em"),
        ("SuperGPQA Easy (mean)", "supergpqa:easy:mcf_em:_average", "em"),
        ("SuperGPQA Middle (mean)", "supergpqa:middle:mcf_em:_average", "em"),
        ("SuperGPQA Hard (mean)", "supergpqa:hard:mcf_em:_average", "em"),
        ("RBench English", "rbench:en:mcf_em", "em"),
        ("MMLU-CF (mean)", "mmlu_cf:mcf_em:_average", "em"),
        ("BIG-Bench Hard (mean)", "bigbench_hard:mcf_em:_average", "em"),
        ("QASC", "qasc:mcf_em", "em"),
        ("MedMCQA", "med_mcqa:mcf_em", "em"),
    ),
}
METRIC_LABELS = {"em": "EM", "f1": "F1", "acc": "Accuracy"}


def normalize_model(model: str) -> str:
    return os.path.normpath(os.path.expanduser(model.strip()))


def safe_name(model: str) -> str:
    """Match the directory naming, including truncation, in scripts/eval.py."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip().strip("/"))[:180]


def resolve_model_dir(model: str, results_dir: Path) -> Path:
    exact = results_dir / safe_name(model)
    if "/" in model:
        return exact
    if not results_dir.is_dir():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    name = safe_name(model)
    matches = sorted(
        path for path in results_dir.iterdir()
        if path.is_dir() and (path.name == name or path.name.endswith("_" + name))
    )
    if len(matches) > 1:
        paths = "\n  ".join(str(path) for path in matches)
        raise ValueError(
            f"Ambiguous model name {model!r}; matching result directories:\n  {paths}\n"
            "Pass the full original model path to disambiguate."
        )
    return matches[0] if matches else exact


def model_matches(model: str, recorded_model: str) -> bool:
    if not isinstance(recorded_model, str):
        return False
    recorded = normalize_model(recorded_model)
    return recorded == model or ("/" not in model and recorded.rsplit("/", 1)[-1] == model)


def result_timestamp(path: Path) -> datetime:
    try:
        timestamp = datetime.fromisoformat(path.stem.removeprefix("results_"))
    except ValueError as exc:
        raise ValueError(f"Cannot select the latest run: invalid timestamp in {path}") from exc
    # Saved timestamps generally have no offset. Use a consistent timezone for
    # ordering, without relying on local machine time or filesystem mtimes.
    return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp


def choose_result(candidates: list[tuple[float | None, Path]], policy: str, context: str):
    if len(candidates) == 1:
        return candidates[0]
    paths = "\n  ".join(str(path) for _, path in candidates)
    if policy == "error":
        raise ValueError(
            f"Multiple saved results for {context}:\n  {paths}\n"
            "Use --duplicates latest to choose by saved timestamp."
        )
    dated = [(result_timestamp(path), value, path) for value, path in candidates]
    latest_time = max(timestamp for timestamp, _, _ in dated)
    latest = [(value, path) for timestamp, value, path in dated if timestamp == latest_time]
    if len(latest) != 1:
        raise ValueError(f"Latest timestamp is tied for {context}:\n  {paths}")
    return latest[0]


def load_results(path: Path, model: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"Missing or invalid results object in {path}")
    config = payload.get("config_general", {})
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config_general in {path}")
    recorded_model = config.get("model_name")
    if recorded_model is None:
        model_config = config.get("model_config", {})
        if isinstance(model_config, dict):
            recorded_model = model_config.get("model_name")
    if recorded_model is not None and not model_matches(model, recorded_model):
        raise ValueError(f"Model mismatch in {path}: expected {model!r}, found {recorded_model!r}")
    return payload["results"]


def metric_value(metrics: dict, task: str, metric: str, path: Path) -> float | None:
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid metrics for {task} in {path}")
    if metric not in metrics:
        return None
    value = metrics[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid score for {task}/{metric} in {path}: {value!r}")
    if not 0 <= value <= 1:
        raise ValueError(f"Expected a 0-1 score for {task}/{metric} in {path}, found {value!r}")
    return value


def read_model_scores(model: str, results_dir: Path, requested: dict[str, str], policy: str):
    model_dir = resolve_model_dir(model, results_dir)
    files = sorted(model_dir.rglob("results_*.json"))
    if not files:
        raise ValueError(f"No results_*.json files for {model}; searched {model_dir}")
    candidates = {task: [] for task in requested}
    for path in files:
        results = load_results(path, model)
        for task, metric in requested.items():
            if task in results:
                candidates[task].append((metric_value(results[task], task, metric, path), path))
    selected = {}
    warnings = []
    for task, records in candidates.items():
        if not records:
            continue
        record = choose_result(records, policy, f"{model}: {task}/{requested[task]}")
        if record[0] is not None:
            selected[task] = record
        if len(records) > 1:
            warnings.append(f"{model}: {task}: selected {record[1]} from {len(records)} runs")
    missing = [f"{task}/{metric}" for task, metric in requested.items() if task not in selected]
    if missing:
        warnings.append(f"{model}: missing {len(missing)} scores (blank cells): {', '.join(missing)}")
    return selected, missing, warnings


def score_std(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return statistics.pstdev(values)


def score_ranks(values: list[float | None]) -> list[float | None]:
    """Rank all compared models, giving tied scores their mean position."""
    if any(value is None for value in values):
        return [None] * len(values)
    positions = {}
    for position, value in enumerate(sorted(values, reverse=True), start=1):
        positions.setdefault(value, []).append(position)
    return [statistics.mean(positions[value]) for value in values]


def build_rows(models: list[str], scores: dict, fewshot: int, sort: str) -> list[dict]:
    rows = []
    for category, specs in CATEGORIES.items():
        benchmarks = []
        for label, task, metric in specs:
            task_key = f"{task}|{fewshot}"
            values = [scores[model][task_key][0] if task_key in scores[model] else None for model in models]
            benchmarks.append({
                "category": category,
                "task": label,
                "metric": METRIC_LABELS[metric],
                "task_key": task_key,
                "values": values,
                "std": score_std(values),
                "row_type": "benchmark",
            })
        if sort == "std":
            benchmarks.sort(key=lambda row: (row["std"] is None, -(row["std"] or 0)))
        else:
            benchmarks.sort(key=lambda row: row["task"].casefold())
        rows.extend(benchmarks)
        aggregate = []
        average_ranks = []
        benchmark_ranks = [score_ranks(row["values"]) for row in benchmarks]
        for index in range(len(models)):
            values = [row["values"][index] for row in benchmarks]
            aggregate.append(None if any(value is None for value in values) else statistics.mean(values))
            ranks = [task_ranks[index] for task_ranks in benchmark_ranks]
            average_ranks.append(None if any(rank is None for rank in ranks) else statistics.mean(ranks))
        rows.append({
            "category": category,
            "task": f"{category} aggregate ({len(benchmarks)} tasks)",
            "metric": "Equal-weight macro mean (mixed EM/F1/Accuracy)",
            "task_key": category.lower().replace(" ", "_") + ":macro_average",
            "values": aggregate,
            "std": score_std(aggregate),
            "row_type": "aggregate",
        })
        rows.append({
            "category": category,
            "task": f"{category} average rank ({len(benchmarks)} tasks)",
            "metric": "Mean task rank (lower is better)",
            "task_key": category.lower().replace(" ", "_") + ":average_rank",
            "values": average_ranks,
            "std": None,
            "row_type": "average_rank",
        })
    return rows


def write_csv(output: Path, models: list[str], rows: list[dict], raw: bool):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "task", "metric", "task_key", *models, "std" if raw else "std_pp", "row_type"])
        for row in rows:
            scale = 1 if raw or row["row_type"] == "average_rank" else 100
            values = ["" if value is None else f"{value * scale:.2f}" for value in row["values"]]
            writer.writerow([
                row["category"], row["task"], row["metric"], row["task_key"],
                *values, "" if row["std"] is None else f"{row['std'] * scale:.2f}", row["row_type"],
            ])


def short_name(model: str) -> str:
    name = model.rsplit("/", 1)[-1]
    aliases = {
        "NVIDIA-Nemotron-3.5": "Nemotron 3.5",
        "NVIDIA-Nemotron-3-Nano": "Nemotron 3 Nano",
        "Trinity-Mini": "Trinity Mini",
        "granite-4.0": "Granite 4.0 H",
        "LFM2.5": "LFM2.5",
        "Mellum2": "Mellum2",
        "gemma-4": "Gemma 4",
    }
    return next((alias for prefix, alias in aliases.items() if name.lower().startswith(prefix.lower())), name)


def markdown_cell(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;"
    ).replace("|", "&#124;").replace("\r", " ").replace("\n", " ")


def render_markdown(models: list[str], rows: list[dict], raw: bool) -> str:
    unit = "fractions (0-1)" if raw else "percentages (0-100)"
    std_label = "Std" if raw else "Std (pp)"
    lines = [
        "# Model score comparison", "",
        f"Scores are {unit}. {std_label} is the population standard deviation across all requested models.", "",
        "Category aggregates are equal-weight means of all listed tasks. Missing scores, aggregates with "
        "missing tasks, and standard deviations with missing model scores are blank.", "",
        "Category average ranks are means of per-task model ranks (1 is best; lower is better). "
        "Ties share their average rank. Ranks stay unscaled and are blank if any requested model "
        "is missing a task in the category.",
    ]
    names = [short_name(model) for model in models]
    # Keep model identities distinct when aliases or final path components collide.
    names = [name if names.count(name) == 1 else model for name, model in zip(names, models)]
    for category in CATEGORIES:
        group = [row for row in rows if row["category"] == category]
        count = sum(row["row_type"] == "benchmark" for row in group)
        lines.extend(["", f"## {markdown_cell(category)} ({count} tasks)", ""])
        heads = ["Task", "Metric", *[markdown_cell(name) for name in names], std_label]
        lines.append("| " + " | ".join(heads) + " |")
        lines.append("| :--- | :--- | " + " | ".join(["---:"] * (len(models) + 1)) + " |")
        for row in group:
            scale = 1 if raw or row["row_type"] == "average_rank" else 100
            label = markdown_cell(row["task"])
            if row["row_type"] in ("aggregate", "average_rank"):
                label = f"**{label}**"
            values = ["" if value is None else f"{value * scale:.2f}" for value in row["values"]]
            std = "" if row["std"] is None else f"{row['std'] * scale:.2f}"
            lines.append("| " + " | ".join([label, row["metric"], *values, std]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="Exact model names, original model paths, or Hugging Face IDs")
    parser.add_argument("--results-dir", type=Path, default=Path.home() / "results", help="Results root (default: ~/results)")
    parser.add_argument("-o", "--output", type=Path, default=Path("model_comparison.csv"), help="CSV path (default: model_comparison.csv)")
    parser.add_argument("--markdown-output", type=Path, help="Markdown path (default: CSV path with .md suffix)")
    parser.add_argument("--fewshot", type=int, default=5, help="Requested few-shot suffix (default: 5)")
    units = parser.add_mutually_exclusive_group()
    units.add_argument("--raw", action="store_true", help="Write 0-1 scores in both reports (default)")
    units.add_argument("--percentages", dest="raw", action="store_false", help="Write percentage scores (0-100)")
    parser.set_defaults(raw=True)
    parser.add_argument("--strict", action="store_true", help="Fail before writing either report if any selected score is missing")
    parser.add_argument("--duplicates", choices=("error", "latest"), default="error", help="Repeated task results (default: error; latest uses filename timestamps)")
    parser.add_argument("--sort", choices=("std", "task"), default="std", help="Order within each category: descending standard deviation or task name (default: std)")
    args = parser.parse_args(argv)
    if args.fewshot < 0:
        parser.error("--fewshot must be nonnegative")
    if any(not model.strip() for model in args.models):
        parser.error("Model names or paths must not be empty")
    models = [normalize_model(model) for model in args.models]
    if len(models) != len(set(models)):
        parser.error("Model names or paths must be unique")
    output = args.output.expanduser()
    markdown_output = args.markdown_output.expanduser() if args.markdown_output else output.with_suffix(".md")
    if output.resolve() == markdown_output.resolve():
        parser.error("CSV and Markdown output paths must be different")
    requested = {f"{task}|{args.fewshot}": metric for specs in CATEGORIES.values() for _, task, metric in specs}
    try:
        scores = {}
        incomplete = False
        for model in models:
            selected, missing, warnings = read_model_scores(model, args.results_dir.expanduser(), requested, args.duplicates)
            scores[model] = selected
            incomplete |= bool(missing)
            for warning in warnings:
                print(f"warning: {warning}", file=sys.stderr)
        if args.strict and incomplete:
            raise ValueError("Missing selected scores; --strict prevented writing either report")
        rows = build_rows(models, scores, args.fewshot, args.sort)
        markdown = render_markdown(models, rows, args.raw)
        write_csv(output, models, rows, args.raw)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(markdown, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    unit = "0-1 scores" if args.raw else "percentages"
    print(f"Wrote {len(requested)} tasks in {len(CATEGORIES)} categories for {len(models)} models ({unit}):")
    print(f"  CSV: {output}")
    print(f"  Markdown: {markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
