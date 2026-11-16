"""CPU-only checks for exporting saved evaluation scores."""

import contextlib
import csv
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_scores.py"
_SPEC = importlib.util.spec_from_file_location("export_scores", _SCRIPT)
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


class TestExportScores(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "comparison.csv"

    def write_run(self, model, results, timestamp="2026-11-15T12:00:00", folder="suite/hash", recorded_model=None):
        # Independent of the exporter's directory-name helper.
        model_dir = {"/models/A": "models_A", "/models/B": "models_B"}[model]
        path = self.root / model_dir / folder / "results" / f"results_{timestamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_general": {"model_name": model if recorded_model is None else recorded_model},
            "results": results,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def run_export(self, *models, extra=()):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = exporter.main([
                *models, "--results-dir", str(self.root), "--output", str(self.output), *extra,
            ])
        return status, stdout.getvalue(), stderr.getvalue()

    def rows(self):
        with self.output.open(newline="") as stream:
            return {row["task_key"]: row for row in csv.DictReader(stream)}

    def test_multiple_models_use_all_requested_metrics_and_saved_means(self):
        f1_tasks = [
            "coqa:gen|5", "drop:gen|5", "natural_questions:gen|5", "popqa:gen|5",
            "squad_v2:gen|5", "triviaqa:gen|5", "truthfulqa:gen|5", "wikifact:gen:_average|5",
        ]
        results = {task: {"em": 0.1, "f1": 0.8} for task in f1_tasks}
        results.update({
            "jeopardy_mc:mcf|5": {"acc": 0.7, "acc_norm": 0.9},
            "mmlu:mcf_em:_average|5": {"em": 0.73},
            "mmlu:anatomy:mcf_em|5": {"em": 1.0},
            "hellaswag:mcf_em|5": {"em": 0.6, "f1": 0.9},
            "all": {"em": 0.55, "f1": 0.95},
        })
        self.write_run("/models/A", results)
        self.write_run("/models/B", {"coqa:gen|5": {"em": 0.3, "f1": 0.4}})
        status, _, stderr = self.run_export("/models/A/", "/models/B")
        self.assertEqual(status, 0, stderr)
        rows = self.rows()
        self.assertEqual(len(rows), 30)
        for task in f1_tasks:
            self.assertEqual(rows[task]["metric"], "F1")
            self.assertEqual(float(rows[task]["/models/A"]), 80)
        self.assertEqual(rows["jeopardy_mc:mcf|5"]["metric"], "Accuracy")
        self.assertEqual(float(rows["jeopardy_mc:mcf|5"]["/models/A"]), 70)
        self.assertEqual(float(rows["mmlu:mcf_em:_average|5"]["/models/A"]), 73)
        self.assertEqual(float(rows["hellaswag:mcf_em|5"]["/models/A"]), 60)
        self.assertNotIn("all", rows)
        self.assertEqual(float(rows["coqa:gen|5"]["/models/B"]), 40)
        self.assertEqual(rows["mmlu:mcf_em:_average|5"]["/models/B"], "")

    def test_exact_model_names_work_without_full_paths(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}})
        self.write_run("/models/B", {"coqa:gen|5": {"f1": 0.4}})
        status, _, stderr = self.run_export("A", "B")
        self.assertEqual(status, 0, stderr)
        self.assertEqual(float(self.rows()["coqa:gen|5"]["A"]), 80)
        self.assertEqual(float(self.rows()["coqa:gen|5"]["B"]), 40)

    def test_ambiguous_model_names_require_full_paths(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}})
        other = self.root / "other_models_A" / "suite" / "results_2026-11-15T12:00:00.json"
        other.parent.mkdir(parents=True)
        other.write_text(json.dumps({
            "config_general": {"model_name": "/other/models/A"},
            "results": {"coqa:gen|5": {"f1": 0.9}},
        }))
        status, _, stderr = self.run_export("A")
        self.assertEqual(status, 2)
        self.assertIn("Ambiguous model name", stderr)
        self.assertIn("models_A", stderr)
        self.assertIn("other_models_A", stderr)
        self.assertFalse(self.output.exists())
        status, _, stderr = self.run_export("/models/A")
        self.assertEqual(status, 0, stderr)
        self.assertEqual(float(self.rows()["coqa:gen|5"]["/models/A"]), 80)

    def test_short_name_still_checks_saved_model_identity(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}}, recorded_model="/models/B")
        status, _, stderr = self.run_export("A")
        self.assertEqual(status, 2)
        self.assertIn("Model mismatch", stderr)
        self.assertFalse(self.output.exists())

    def test_disjoint_task_files_are_combined(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}}, folder="coqa/hash")
        self.write_run("/models/A", {"hellaswag:mcf_em|5": {"em": 0.6}}, folder="hellaswag/hash")
        status, _, stderr = self.run_export("/models/A")
        self.assertEqual(status, 0, stderr)
        self.assertEqual(float(self.rows()["coqa:gen|5"]["/models/A"]), 80)
        self.assertEqual(float(self.rows()["hellaswag:mcf_em|5"]["/models/A"]), 60)

    def test_duplicate_runs_require_explicit_resolution_and_use_timestamp(self):
        older = self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}})
        newer = self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.9}}, "2026-11-16T12:00:00")
        os.utime(older, (2_000_000_000, 2_000_000_000))
        os.utime(newer, (1, 1))
        status, _, stderr = self.run_export("/models/A")
        self.assertEqual(status, 2)
        self.assertIn("Multiple saved results", stderr)
        self.assertFalse(self.output.exists())
        status, _, stderr = self.run_export("/models/A", extra=("--duplicates", "latest"))
        self.assertEqual(status, 0, stderr)
        self.assertIn(str(newer), stderr)
        self.assertEqual(float(self.rows()["coqa:gen|5"]["/models/A"]), 90)

    def test_latest_run_missing_metric_does_not_fall_back_to_an_older_score(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}})
        self.write_run("/models/A", {"coqa:gen|5": {"em": 0.6}}, "2026-11-16T12:00:00")
        status, _, stderr = self.run_export("/models/A")
        self.assertEqual(status, 2)
        self.assertIn("Multiple saved results", stderr)
        status, _, stderr = self.run_export("/models/A", extra=("--duplicates", "latest"))
        self.assertEqual(status, 0, stderr)
        self.assertIn("coqa:gen|5/f1", stderr)
        self.assertEqual(self.rows()["coqa:gen|5"]["/models/A"], "")

    def test_latest_timestamp_ties_remain_errors(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}}, folder="first/hash")
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.9}}, folder="second/hash")
        status, _, stderr = self.run_export("/models/A", extra=("--duplicates", "latest"))
        self.assertEqual(status, 2)
        self.assertIn("timestamp is tied", stderr)
        self.assertFalse(self.output.exists())

    def test_missing_metrics_are_blank_zero_is_preserved_and_strict_does_not_overwrite(self):
        self.write_run("/models/A", {"coqa:gen|5": {"em": 0.8}, "hellaswag:mcf_em|5": {"em": 0}})
        status, _, stderr = self.run_export("/models/A")
        self.assertEqual(status, 0, stderr)
        self.assertIn("coqa:gen|5/f1", stderr)
        self.assertEqual(self.rows()["coqa:gen|5"]["/models/A"], "")
        self.assertEqual(float(self.rows()["hellaswag:mcf_em|5"]["/models/A"]), 0)
        self.output.write_text("keep existing output\n")
        status, _, stderr = self.run_export("/models/A", extra=("--strict",))
        self.assertEqual(status, 2)
        self.assertIn("--strict prevented", stderr)
        self.assertEqual(self.output.read_text(), "keep existing output\n")

    def test_raw_and_fewshot_selection_do_not_mix_variants(self):
        self.write_run("/models/A", {"coqa:gen|0": {"f1": 0.825}, "coqa:gen|5": {"f1": 0.5}})
        status, _, stderr = self.run_export("/models/A", extra=("--raw", "--fewshot", "0"))
        self.assertEqual(status, 0, stderr)
        self.assertEqual(float(self.rows()["coqa:gen|0"]["/models/A"]), 0.825)
        self.assertNotIn("coqa:gen|5", self.rows())

    def test_model_identity_mismatch_is_an_error(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}}, recorded_model="/models/B")
        status, _, stderr = self.run_export("/models/A")
        self.assertEqual(status, 2)
        self.assertIn("Model mismatch", stderr)
        self.assertFalse(self.output.exists())

    def test_invalid_scores_are_errors(self):
        for value in (float("nan"), float("inf"), True, "0.8", -0.1, 80):
            with self.subTest(value=value):
                self.write_run("/models/A", {"coqa:gen|5": {"f1": value}})
                status, _, stderr = self.run_export("/models/A")
                self.assertEqual(status, 2)
                self.assertIn("score", stderr)
                self.assertFalse(self.output.exists())

    def test_unknown_model_is_an_error(self):
        status, _, stderr = self.run_export("/models/A")
        self.assertEqual(status, 2)
        self.assertIn("No results_*.json", stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
