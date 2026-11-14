"""Tests of zebralogic_chat (chat_zebra.py): the official prompt, the JSON extraction and the cell / puzzle scoring, the size buckets.

Plain functions: run with pytest, or `python tests/unit/tasks/test_chat_zebra.py`. The last test reads the pinned private dataset from the
Hugging Face cache and skips when it is not there (the gated set: the owner accepted the terms on 11-14)."""

import json
import unittest

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import chat_zebra as z


SOLUTION = {
    "header": ["House", "Name", "Drink"],
    "rows": [["1", "Arnold", "tea"], ["2", "Peter", "water"], ["3", "Eric", "milk"]],
}
TABLE = {"House 1": {"Name": "Arnold", "Drink": "tea"}, "House 2": {"Name": "Peter", "Drink": "water"}, "House 3": {"Name": "Eric", "Drink": "milk"}}


def answer(table, reasoning="r"):
    return json.dumps({"reasoning": reasoning, "solution": table})


def run_metric(size, text, table=TABLE):
    doc = Doc(task_name="zebralogic_chat", query="q", choices=[""], gold_index=0, specific={"solution_table": table, "size": size, "id": "x"})
    return z.ZebraChatMetrics().compute(doc, ModelResponse(text=[text]))


class ZebraChatTest(unittest.TestCase):
    def test_prompt_is_the_official_template(self):
        line = {"id": "x", "size": "3*2", "puzzle": "PUZZLE TEXT", "solution": SOLUTION}
        doc = z.zebra_chat_prompt(line, "zebralogic_chat")
        self.assertIn("# Puzzle to Solve \n\nPUZZLE TEXT", doc.query)
        self.assertNotIn("{puzzle}", doc.query)
        self.assertNotIn("{json_template}", doc.query)
        self.assertIn('"House 1"', doc.query)
        self.assertIn('"Drink": "___"', doc.query)
        self.assertNotIn('"House": ', doc.query)                 # the House column is not an attribute
        self.assertEqual(doc.specific["solution_table"], TABLE)

    def test_extract_last_complete_json(self):
        self.assertEqual(z.extract_last_complete_json('x {"a": {"b": 1}} y {"c": 2}'), {"c": 2})
        self.assertEqual(z.extract_last_complete_json('{"solution": {"House 1": {"Name":\n"A"}}}'), {"solution": {"House 1": {"Name": "A"}}})
        self.assertIsNone(z.extract_last_complete_json("no json {unclosed"))
        self.assertIsNone(z.extract_last_complete_json('{"a": 1,}'))

    def test_correct_answer_scores_puzzle_and_cells(self):
        s = run_metric("3*2", answer(TABLE))
        self.assertEqual((s["puzzle_acc"], s["cell_acc"], s["parsed"]), (1.0, 1.0, 1.0))

    def test_case_and_whitespace_insensitive(self):
        upper = {h: {k: v.upper() + " " for k, v in cells.items()} for h, cells in TABLE.items()}
        self.assertEqual(run_metric("3*2", answer(upper))["puzzle_acc"], 1.0)

    def test_one_wrong_cell_fails_the_puzzle_and_counts_cells(self):
        wrong = json.loads(answer(TABLE))
        wrong["solution"]["House 1"]["Drink"] = "milk"
        s = run_metric("3*2", json.dumps(wrong))
        self.assertEqual(s["puzzle_acc"], 0.0)
        self.assertAlmostEqual(s["cell_acc"], 5 / 6)

    def test_list_value_uses_first_item_and_bad_value_is_wrong(self):
        table = {h: dict(cells) for h, cells in TABLE.items()}
        table["House 1"]["Name"] = ["Arnold"]
        table["House 2"]["Drink"] = 5
        s = run_metric("3*2", answer(table))
        self.assertAlmostEqual(s["cell_acc"], 5 / 6)     # the list cell is right, the int cell is wrong

    def test_no_answer_scores_zero_and_is_not_parsed(self):
        for text in ["", "I could not solve it", "reasoning only {not json", answer(TABLE).replace('"solution"', '"answer"')]:
            s = run_metric("3*2", text)
            self.assertEqual((s["puzzle_acc"], s["cell_acc"], s["parsed"]), (0.0, 0.0, 0.0), text)

    def test_size_buckets(self):
        self.assertEqual(run_metric("2*2", answer(TABLE))["small_acc"], 1.0)
        s = run_metric("2*2", answer(TABLE))
        self.assertIsNone(s["xl_acc"])
        self.assertIsNone(s["medium_acc"])
        self.assertEqual(run_metric("6*6", "")["xl_acc"], 0.0)
        self.assertEqual(run_metric("4*5", answer(TABLE))["large_acc"], 1.0)
        self.assertEqual(run_metric("3*5", answer(TABLE))["medium_acc"], 1.0)

    def test_buckets_cover_the_25_sizes(self):
        self.assertEqual(len(z.BUCKET_OF), 25)
        self.assertEqual(sorted(len(v) for v in z.SIZE_BUCKETS.values()), [5, 5, 7, 8])

    def test_official_grids_score_as_correct(self):
        try:
            import pyarrow.parquet as pq
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            self.skipTest("pyarrow / huggingface_hub not installed")
        path = try_to_load_from_cache("allenai/ZebraLogicBench-private", "grid_mode/test-00000-of-00001.parquet", revision=z.ZEBRA_REVISION, repo_type="dataset")
        if not isinstance(path, str):
            self.skipTest("the private set is not in the Hugging Face cache")
        rows = pq.read_table(path).to_pylist()[:20]
        for row in rows:
            header = row["solution"]["header"]
            table = {f"House {i + 1}": {header[j]: r[j] for j in range(1, len(header))} for i, r in enumerate(row["solution"]["rows"])}
            s = run_metric(row["size"], answer(table), table)
            self.assertEqual(s["puzzle_acc"], 1.0, row["id"])


if __name__ == "__main__":
    unittest.main()
