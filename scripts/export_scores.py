r"""Export the selected base-suite metrics for multiple saved model evaluations.

Example:
    python scripts/export_scores.py model-a model-b \
        --results-dir ~/results --output results/model_comparison.csv

The CSV has one row per task and one score column per model. Scores are
percentages by default; --raw preserves the saved 0-1 values. Group means are
read from the saved results, never reconstructed from incomplete subtasks.

Accept full model paths, Hugging Face IDs, or exact model names (the final path
component). Names must match a single result directory; ambiguous names produce
an error listing the matches. Full paths follow scripts/eval.py's result-directory
naming convention. Model weights need not exist locally.
Missing scores produce blank cells and warnings;
--strict makes them an error. Repeated task results are errors unless explicitly
resolved with --duplicates latest, using timestamps in results filenames.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# Display name, exact saved task name without its few-shot suffix, metric key.
TASK_METRICS = (
    ("AGIEval English (mean)", "agieval_eng:mcf_em:_average", "em"),
    ("ARC Challenge", "arc:challenge:mcf_em", "em"),
    ("ARC Easy", "arc:easy:mcf_em", "em"),
    ("ARC (mean)", "arc:mcf_em:_average", "em"),
    ("BIG-Bench Hard (mean)", "bigbench_hard:mcf_em:_average", "em"),
    ("CommonsenseQA", "commonsenseqa:mcf_em", "em"),
    ("CoQA", "coqa:gen", "f1"),
    ("DROP", "drop:gen", "f1"),
    ("HellaSwag", "hellaswag:mcf_em", "em"),
    ("Jeopardy MC", "jeopardy_mc:mcf", "acc"),
    ("MedMCQA", "med_mcqa:mcf_em", "em"),
    ("MMLU (mean)", "mmlu:mcf_em:_average", "em"),
    ("MMLU-CF (mean)", "mmlu_cf:mcf_em:_average", "em"),
    ("Natural Questions", "natural_questions:gen", "f1"),
    ("OpenBookQA", "openbookqa:mcf_em", "em"),
    ("PIQA", "piqa:mcf_em", "em"),
    ("PopQA", "popqa:gen", "f1"),
    ("QASC", "qasc:mcf_em", "em"),
    ("RBench English", "rbench:en:mcf_em", "em"),
    ("SciQ", "sciq:mcf_em", "em"),
    ("SIQA", "siqa:mcf_em", "em"),
    ("SQuAD v2 (answerable only)", "squad_v2:gen", "f1"),
    ("SuperGPQA (discipline mean)", "supergpqa:mcf_em:_average", "em"),
    ("SuperGPQA Easy (mean)", "supergpqa:easy:mcf_em:_average", "em"),
    ("SuperGPQA Middle (mean)", "supergpqa:middle:mcf_em:_average", "em"),
    ("SuperGPQA Hard (mean)", "supergpqa:hard:mcf_em:_average", "em"),
    ("TriviaQA", "triviaqa:gen", "f1"),
    ("TruthfulQA (generative)", "truthfulqa:gen", "f1"),
    ("WikiFact (mean)", "wikifact:gen:_average", "f1"),
    ("WinoGrande", "winogrande:mcf_em", "em"),
)
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


def write_csv(output: Path, models: list[str], specs: list[tuple[str, str, str]], scores: dict, raw: bool):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "metric", "task_key", *models])
        for label, task, metric in specs:
            values = []
            for model in models:
                record = scores[model].get(task)
                values.append("" if record is None else record[0] if raw else record[0] * 100)
            writer.writerow([label, METRIC_LABELS[metric], task, *values])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="Exact model names, original model paths, or Hugging Face IDs")
    parser.add_argument("--results-dir", type=Path, default=Path.home() / "results", help="Results root (default: ~/results)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Destination CSV")
    parser.add_argument("--fewshot", type=int, default=5, help="Requested few-shot suffix to select (default: 5)")
    parser.add_argument("--raw", action="store_true", help="Write saved 0-1 scores instead of percentages")
    parser.add_argument("--strict", action="store_true", help="Fail before writing if any selected score is missing")
    parser.add_argument(
        "--duplicates",
        choices=("error", "latest"),
        default="error",
        help="How to handle repeated task results (default: error; latest uses filename timestamps)",
    )
    args = parser.parse_args(argv)
    if args.fewshot < 0:
        parser.error("--fewshot must be nonnegative")
    if any(not model.strip() for model in args.models):
        parser.error("Model names or paths must not be empty")
    models = [normalize_model(model) for model in args.models]
    if len(models) != len(set(models)):
        parser.error("Model names or paths must be unique")
    specs = [(label, task if task == "all" else f"{task}|{args.fewshot}", metric) for label, task, metric in TASK_METRICS]
    requested = {task: metric for _, task, metric in specs}
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
            raise ValueError("Missing selected scores; --strict prevented writing the CSV")
        output = args.output.expanduser()
        write_csv(output, models, specs, scores, args.raw)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    unit = "0-1 scores" if args.raw else "percentages"
    print(f"Wrote {len(specs)} tasks for {len(models)} models to {output} ({unit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
