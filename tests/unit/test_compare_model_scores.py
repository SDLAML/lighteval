"""CPU-only checks for grouped JSON-to-CSV/Markdown score comparisons."""

import contextlib
import csv
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_model_scores.py"
_SPEC = importlib.util.spec_from_file_location("compare_model_scores", _SCRIPT)
comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison)


class TestCompareModelScores(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "reports" / "comparison.csv"
        self.markdown = self.output.with_suffix(".md")

    def write_run(self, model, results, timestamp="2026-11-15T12:00:00", folder="suite/hash", recorded_model=None):
        directory = {"/models/A": "models_A", "/models/B": "models_B", "/other/A": "other_A"}[model]
        path = self.root / directory / folder / "results" / f"results_{timestamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "config_general": {"model_name": model if recorded_model is None else recorded_model},
            "results": results,
        }), encoding="utf-8")
        return path

    def run_comparison(self, *models, extra=()):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = comparison.main([
                *models, "--results-dir", str(self.root), "--output", str(self.output), *extra,
            ])
        return status, stdout.getvalue(), stderr.getvalue()

    def rows(self):
        with self.output.open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    def complete_results(self, general=0.2, scientific=0.3, fewshot=5):
        return {
            f"{task}|{fewshot}": {metric: general if category == "General QA" else scientific}
            for category, specs in comparison.CATEGORIES.items()
            for _, task, metric in specs
        }

    def test_complete_reports_select_metrics_saved_means_and_equal_weight_aggregates(self):
        first = self.complete_results()
        second = self.complete_results(general=0.6, scientific=0.7)
        first["hellaswag:mcf_em|5"] = {"em": 0, "f1": 0.99}
        second["hellaswag:mcf_em|5"] = {"em": 1}
        first["mmlu:anatomy:mcf_em|5"] = {"em": 1}
        first["arc:mcf_em:_average|5"] = {"em": 1}
        first["supergpqa:mcf_em:_average|5"] = {"em": 1}
        self.write_run("/models/A", first)
        self.write_run("/models/B", second)
        status, _, stderr = self.run_comparison("/models/A", "/models/B", extra=("--strict",))
        self.assertEqual(status, 0, stderr)
        self.assertEqual(stderr, "")
        rows = self.rows()
        self.assertEqual(len(rows), 30)
        benchmarks = [row for row in rows if row["row_type"] == "benchmark"]
        self.assertEqual(sum(row["category"] == "General QA" for row in benchmarks), 15)
        self.assertEqual(sum(row["category"] == "Scientific QA" for row in benchmarks), 13)
        by_key = {row["task_key"]: row for row in rows}
        self.assertEqual(rows[0]["task_key"], "hellaswag:mcf_em|5")
        self.assertEqual(float(rows[0]["std_pp"]), 50)
        self.assertEqual(float(rows[0]["/models/A"]), 0)
        self.assertEqual(by_key["coqa:gen|5"]["metric"], "F1")
        self.assertEqual(by_key["jeopardy_mc:mcf|5"]["metric"], "Accuracy")
        self.assertEqual(float(by_key["mmlu:mcf_em:_average|5"]["/models/A"]), 30)
        self.assertNotIn("arc:mcf_em:_average|5", by_key)
        self.assertNotIn("supergpqa:mcf_em:_average|5", by_key)
        general = by_key["general_qa:macro_average"]
        self.assertAlmostEqual(float(general["/models/A"]), 100 * 14 * 0.2 / 15)
        self.assertAlmostEqual(float(general["/models/B"]), 100 * (14 * 0.6 + 1) / 15)
        self.assertAlmostEqual(float(general["std_pp"]), (float(general["/models/B"]) - float(general["/models/A"])) / 2)
        self.assertAlmostEqual(float(by_key["scientific_qa:macro_average"]["/models/B"]), 70)
        markdown = self.markdown.read_text()
        self.assertIn("## General QA (15 tasks)", markdown)
        self.assertIn("## Scientific QA (13 tasks)", markdown)
        self.assertIn("| HellaSwag | EM | 0.00 | 100.00 | 50.00 |", markdown)
        self.assertIn("| MMLU (mean) | EM | 30.00 | 70.00 | 20.00 |", markdown)
        self.assertIn("**General QA aggregate (15 tasks)**", markdown)
        self.assertIn("percentages (0-100)", markdown)

    def test_missing_scores_keep_zero_and_do_not_create_partial_aggregates(self):
        self.write_run("/models/A", self.complete_results())
        self.write_run("/models/B", {"hellaswag:mcf_em|5": {"em": 0}})
        status, _, stderr = self.run_comparison("A", "B")
        self.assertEqual(status, 0, stderr)
        self.assertIn("missing 27 scores", stderr)
        rows = {row["task_key"]: row for row in self.rows()}
        self.assertEqual(float(rows["hellaswag:mcf_em|5"]["B"]), 0)
        self.assertEqual(rows["coqa:gen|5"]["B"], "")
        self.assertEqual(rows["coqa:gen|5"]["std_pp"], "")
        self.assertEqual(rows["general_qa:macro_average"]["B"], "")
        self.assertEqual(rows["general_qa:macro_average"]["std_pp"], "")
        self.assertEqual(float(rows["general_qa:macro_average"]["A"]), 20)
        self.assertIn("| CoQA | F1 | 20.00 |  |  |", self.markdown.read_text())

    def test_strict_preserves_both_existing_outputs(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}})
        self.output.parent.mkdir(parents=True)
        self.output.write_text("existing csv")
        self.markdown.write_text("existing markdown")
        status, _, stderr = self.run_comparison("A", extra=("--strict",))
        self.assertEqual(status, 2)
        self.assertIn("--strict prevented", stderr)
        self.assertEqual(self.output.read_text(), "existing csv")
        self.assertEqual(self.markdown.read_text(), "existing markdown")

    def test_latest_uses_filename_timestamp_and_does_not_restore_missing_metrics(self):
        older = self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}, "hellaswag:mcf_em|5": {"em": 0.1}})
        newer = self.write_run("/models/A", {"coqa:gen|5": {"em": 0.9}, "hellaswag:mcf_em|5": {"em": 0.9}}, "2026-11-16T12:00:00")
        os.utime(older, (2_000_000_000, 2_000_000_000))
        os.utime(newer, (1, 1))
        status, _, stderr = self.run_comparison("A")
        self.assertEqual(status, 2)
        self.assertIn("Multiple saved results", stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.markdown.exists())
        status, _, stderr = self.run_comparison("A", extra=("--duplicates", "latest"))
        self.assertEqual(status, 0, stderr)
        self.assertIn(str(newer), stderr)
        rows = {row["task_key"]: row for row in self.rows()}
        self.assertEqual(rows["coqa:gen|5"]["A"], "")
        self.assertEqual(float(rows["hellaswag:mcf_em|5"]["A"]), 90)

    def test_duplicate_timestamp_ties_and_invalid_timestamps_fail(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}}, folder="first")
        repeated = self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.9}}, folder="second")
        status, _, stderr = self.run_comparison("A", extra=("--duplicates", "latest"))
        self.assertEqual(status, 2)
        self.assertIn("timestamp is tied", stderr)
        repeated.rename(repeated.with_name("results_invalid.json"))
        status, _, stderr = self.run_comparison("A", extra=("--duplicates", "latest"))
        self.assertEqual(status, 2)
        self.assertIn("invalid timestamp", stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.markdown.exists())

    def test_raw_fewshot_custom_markdown_and_task_sort(self):
        results = self.complete_results(general=0.825, scientific=0.625, fewshot=0)
        results["coqa:gen|5"] = {"f1": 0.1}
        self.write_run("/models/A", results)
        custom = self.root / "markdown" / "table.md"
        status, stdout, stderr = self.run_comparison("A", extra=(
            "--raw", "--fewshot", "0", "--strict", "--sort", "task", "--markdown-output", str(custom),
        ))
        self.assertEqual(status, 0, stderr)
        rows = self.rows()
        self.assertEqual(rows[0]["task"], "CommonsenseQA")
        coqa = next(row for row in rows if row["task_key"] == "coqa:gen|0")
        self.assertEqual(float(coqa["A"]), 0.825)
        self.assertEqual(float(coqa["std_pp"]), 0)
        self.assertFalse(self.markdown.exists())
        self.assertIn("| CoQA | F1 | 0.8250 | 0.00 |", custom.read_text())
        self.assertIn("fractions (0-1)", custom.read_text())
        self.assertIn("0-1 scores", stdout)

    def test_disjoint_files_are_combined(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}}, folder="coqa/hash")
        self.write_run("/models/A", {"hellaswag:mcf_em|5": {"em": 0.6}}, folder="hellaswag/hash")
        status, _, stderr = self.run_comparison("/models/A/")
        self.assertEqual(status, 0, stderr)
        rows = {row["task_key"]: row for row in self.rows()}
        self.assertEqual(float(rows["coqa:gen|5"]["/models/A"]), 80)
        self.assertEqual(float(rows["hellaswag:mcf_em|5"]["/models/A"]), 60)

    def test_ambiguous_names_fail_and_full_paths_keep_markdown_headers_distinct(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}})
        self.write_run("/other/A", {"coqa:gen|5": {"f1": 0.6}})
        status, _, stderr = self.run_comparison("A")
        self.assertEqual(status, 2)
        self.assertIn("Ambiguous model name", stderr)
        self.assertFalse(self.output.exists())
        status, _, stderr = self.run_comparison("/models/A", "/other/A")
        self.assertEqual(status, 0, stderr)
        self.assertIn("| Task | Metric | /models/A | /other/A | Std (pp) |", self.markdown.read_text())

    def test_model_identity_mismatch_fails_before_writing(self):
        self.write_run("/models/A", {"coqa:gen|5": {"f1": 0.8}}, recorded_model="/models/B")
        status, _, stderr = self.run_comparison("A")
        self.assertEqual(status, 2)
        self.assertIn("Model mismatch", stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.markdown.exists())

    def test_invalid_scores_and_payloads_fail_before_writing(self):
        for value in (float("nan"), float("inf"), True, "0.8", -0.1, 80):
            with self.subTest(value=value):
                self.write_run("/models/A", {"coqa:gen|5": {"f1": value}})
                status, _, stderr = self.run_comparison("A")
                self.assertEqual(status, 2)
                self.assertIn("score", stderr)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.markdown.exists())
        path = self.write_run("/models/A", {})
        for payload in ([], {"results": []}, {"results": {}, "config_general": []}):
            with self.subTest(payload=payload):
                path.write_text(json.dumps(payload))
                status, _, stderr = self.run_comparison("A")
                self.assertEqual(status, 2)
                self.assertIn("invalid", stderr.lower())
                self.assertFalse(self.output.exists())
                self.assertFalse(self.markdown.exists())

    def test_missing_model_fails_before_writing(self):
        status, _, stderr = self.run_comparison("/models/A")
        self.assertEqual(status, 2)
        self.assertIn("No results_*.json", stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.markdown.exists())

    def test_output_collision_and_invalid_arguments_fail_before_writing(self):
        for models, extra in (
            (("A",), ("--markdown-output", str(self.output))),
            (("A",), ("--fewshot", "-1")),
            (("A", "A/"), ()),
            ((" ",), ()),
        ):
            with self.subTest(models=models, extra=extra), self.assertRaises(SystemExit) as raised:
                self.run_comparison(*models, extra=extra)
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(self.output.exists())
            self.assertFalse(self.markdown.exists())

    def test_markdown_escapes_model_names(self):
        markdown = comparison.render_markdown(["org/A|<test>"], [], raw=False)
        self.assertIn("A&#124;&lt;test&gt;", markdown)


if __name__ == "__main__":
    unittest.main()
