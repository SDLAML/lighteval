"""Tests of the base chain-of-thought tasks of 11-16: mmlu_pro_base:cot (5-shot, official TIGER-Lab prompt), bbh_base:*:cot (3-shot, official exemplars) and the
5-shot mlmm_hellaswag:*:mcf_em (held-out demonstrations).

Plain functions without fixtures: run with pytest, or `python tests/unit/tasks/test_base_cot.py`. The tests that read the pinned datasets skip when they are not
in the Hugging Face cache.
"""

import re
import unittest
from pathlib import Path

from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager
from lighteval.tasks.registry import Registry
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks import bbh_base, mmlu_pro_base
from lighteval.tasks.tasks.bbh_cot_demos import BBH_COT_DEMOS
from lighteval.tasks.tasks.bigbench_hard import _BBH_SUBSETS


def _need(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as error:  # dataset not cached / offline
        raise unittest.SkipTest(f"dataset not available: {type(error).__name__}: {str(error)[:120]}")


def _tasks(spec: str, multilingual: bool = False) -> dict:
    return _need(lambda: Registry(tasks=spec, load_multilingual=multilingual).load_tasks())


def _score(metric, doc, text):
    return metric.sample_level_fn.compute(doc, ModelResponse(text=[text]))


# ---- MMLU-Pro ------------------------------------------------------------------------------------------------------------------

def test_mmlu_pro_extraction_follows_the_official_three_levels():
    ex = mmlu_pro_base.extract_answer
    assert ex("blah. The answer is (C).") == ("C", 1) and ex("the answer is D") == ("D", 1) and ex("answer is (J)") == ("J", 1)
    assert ex("Final Answer: B") == ("B", 2) and ex("so the answer: e") == (None, 0)
    assert ex("It is B because x") == ("B", 3) and ex("we pick A and then C") == ("C", 3), "the LAST stand-alone letter A-J"
    assert ex("The answer is (K).") == (None, 0), "K is not an option letter of the script (A-J)"
    assert ex("") == (None, 0) and ex("no letters here at all, only words") == (None, 0)
    assert ex("The answer is (B). Then the answer is (C).") == ("B", 1), "the FIRST match of the first-level rule"


def test_mmlu_pro_metric_scores_and_flags_the_first_level():
    doc = Doc(task_name="t", query="q", choices=list("ABCD"), gold_index=2)
    m = mmlu_pro_base.MMLU_PRO_BASE_METRIC
    assert _score(m, doc, "reasoning. The answer is (C).") == {"acc": 1.0, "answer_is_rate": 1.0}
    assert _score(m, doc, "reasoning. The answer is (B).") == {"acc": 0.0, "answer_is_rate": 1.0}
    assert _score(m, doc, "I think C") == {"acc": 1.0, "answer_is_rate": 0.0}, "a lower-level hit still scores, but is not counted as the requested format"
    assert _score(m, doc, "no letter") == {"acc": 0.0, "answer_is_rate": 0.0}, "no letter at all scores 0 (the script guesses a random option)"


def test_mmlu_pro_gold_is_the_answer_letter_even_when_answer_index_disagrees():
    row = {"question_id": 3983, "question": "q?", "options": ["a", "b", "c", "d", "e", "f", "g"], "answer": "C", "answer_index": 1, "cot_content": "", "category": "chemistry"}
    doc = mmlu_pro_base.mmlu_pro_base_prompt(dict(row), "t")
    assert doc.choices[doc.gold_index] == "C" and doc.gold_index == 2, "the official script compares the letter in `answer`"
    try:
        mmlu_pro_base.mmlu_pro_base_prompt(dict(row, answer="H"), "t")
        raise AssertionError("a gold letter beyond the options must raise")
    except ValueError:
        pass


def test_mmlu_pro_prompt_layout_and_na_options():
    row = {"question_id": 7, "question": "What is 1+1?", "options": ["1", "2", "N/A", "N/A"], "answer": "B", "answer_index": 1, "cot_content": "", "category": "math"}
    doc = mmlu_pro_base.mmlu_pro_base_prompt(dict(row), "t")
    assert doc.query == "Question:\nWhat is 1+1?\nOptions:\nA. 1\nB. 2\nAnswer: Let's think step by step."
    assert doc.choices == ["A", "B"] and doc.gold_index == 1 and doc.specific["fewshot_group"] == "math"
    assert doc.instruction.startswith("The following are multiple choice questions (with answers) about math. Think step by step and then finish")
    assert doc.instruction.endswith('where X is the correct letter choice.\n\n')
    shot = dict(row, cot_content="A: Let's think step by step. 1 plus 1 is 2. The answer is (B).")
    shot["__few_shots"] = True
    sdoc = mmlu_pro_base.mmlu_pro_base_prompt(shot, "t")
    assert sdoc.query == "Question:\nWhat is 1+1?\nOptions:\nA. 1\nB. 2\nAnswer:" and sdoc.choices == ["Let's think step by step. 1 plus 1 is 2. The answer is (B)."]
    bad = dict(shot, cot_content="no prefix")
    try:
        mmlu_pro_base.mmlu_pro_base_prompt(bad, "t")
        raise AssertionError("a shot without the official 'A: Let's think step by step.' prefix must raise")
    except ValueError:
        pass


def test_fewshot_group_keeps_only_the_shots_of_the_evaluated_items_group():
    tasks = _tasks("mmlu_pro_base:cot|5")
    task = tasks["mmlu_pro_base:cot|5"]
    docs = _need(task.get_docs, 60)
    pool = task.fewshot_docs()
    assert len(pool) == 70
    for d in docs:
        assert len(d.fewshot_samples) == 5
        assert {s.specific["fewshot_group"] for s in d.fewshot_samples} == {d.specific["category"]}
        # the first five of the category in dataset order
        want = [s for s in pool if s.specific["fewshot_group"] == d.specific["category"]][:5]
        assert [s.id for s in d.fewshot_samples] == [s.id for s in want]


def _official_prompt(val_rows, row, initial_prompt):
    """generate_cot_prompt / format_cot_example of TIGER-AI-Lab/MMLU-Pro evaluate_from_local.py (f418b116), copied."""
    letters = "ABCDEFGHIJKLMNOP"

    def fmt(example, including_answer=True):
        prompt = "Question:\n" + example["question"] + "\n" + "Options:\n"
        for i, opt in enumerate(example["options"]):
            prompt += "{}. {}\n".format(letters[i], opt)
        if including_answer:
            return prompt + example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.") + "\n\n"
        return prompt + "Answer: Let's think step by step."

    subject = row["category"]
    shots = [e for e in val_rows if e["category"] == subject][:5]
    prompt = initial_prompt.replace("{$}", subject) + "\n"
    for e in shots:
        prompt += fmt(e)
    return prompt + fmt(row, including_answer=False)


def test_mmlu_pro_prompt_is_byte_identical_to_the_official_one():
    initial = ('The following are multiple choice questions (with answers) about {$}. Think step by step and then finish your answer with '
               '"the answer is (X)" where X is the correct letter choice.\n\n\n')            # cot_prompt_lib/initial_prompt.txt (checked with xxd: three newlines)
    task = _tasks("mmlu_pro_base:cot|5")["mmlu_pro_base:cot|5"]
    docs = _need(task.get_docs)
    assert len(docs) == 12032
    val = [dict(r, options=[o for o in r["options"] if o != "N/A"]) for r in _need(lambda: task.dataset["validation"])]
    test = {int(r["question_id"]): dict(r, options=[o for o in r["options"] if o != "N/A"]) for r in task.dataset["test"]}
    pm = PromptManager(use_chat_template=False)
    for d in docs[::37]:                                                                       # about 325 questions of all 14 categories (the full 12,032 were compared by hand)
        assert pm.prepare_prompt(d) == _official_prompt(val, test[d.specific["question_id"]], initial)
    assert {d.specific["category"] for d in docs[::37]} == {r["category"] for r in val}
    assert task.generation_size == 2048 and task.stop_sequence == ["Question:"] and task.num_samples == [1]


# ---- BBH -----------------------------------------------------------------------------------------------------------------------

def test_bbh_demos_cover_the_27_subtasks_with_three_shots_each():
    assert set(BBH_COT_DEMOS) == set(_BBH_SUBSETS) and len(_BBH_SUBSETS) == 27
    seps = {"\n": 0, " ": 0}
    for sub, d in BBH_COT_DEMOS.items():
        assert len(d["shots"]) == 3 and d["description"].strip() == d["description"] and not d["description"].startswith("Q:")
        for s in d["shots"]:
            assert s["answer"].startswith("Let's think step by step.") and s["answer"].endswith(".") and "So the answer is " in s["answer"]
            assert "\n\n" not in s["answer"] and "\n\n" not in s["input"] and "BENCHMARK DATA" not in s["answer"] + s["input"]
        seps[d["shots"][0]["answer"][len("Let's think step by step.")]] += 1
    assert seps == {"\n": 26, " ": 1}


def test_bbh_demos_equal_the_independent_copy_in_lm_evaluation_harness():
    import yaml

    here = Path(__file__).resolve()
    candidates = [here.parents[3].parent / "lm-evaluation-harness", Path("/e/scratch/trustllm-eu/wang55/titan_eval/resources/lm-evaluation-harness")]
    root = next((c / "lm_eval" / "tasks" / "bbh" / "cot_fewshot" for c in candidates if (c / "lm_eval" / "tasks" / "bbh" / "cot_fewshot").is_dir()), None)
    if root is None:
        raise unittest.SkipTest("resources/lm-evaluation-harness not found")
    step = "Let's think step by step."
    for sub, d in BBH_COT_DEMOS.items():
        y = yaml.safe_load((root / f"{sub}.yaml").read_text())
        assert y["description"] == d["description"] + "\n\n", sub
        for s, shot in zip(y["fewshot_config"]["samples"], d["shots"]):
            assert s["input"] == shot["input"] and shot["answer"] == step + shot["answer"][len(step)] + s["target"], sub


def test_bbh_answer_extraction():
    ex = bbh_base.extract_answer
    assert ex("Z = True. So the answer is True.") == "True" and ex("so the answer is (B).\n\nQ: next") == "(B)"
    assert ex("blah. So the answer is (B).\nQ: x") == "(B)" and ex("So the answer is -41.") == "-41" and ex("So the answer is ] ] }.") == "] ] }"
    assert ex("So the answer is no") == "no", "no trailing period needed (lm-eval's regex would cut the last character)"
    assert ex("so the answer is yes.\nAgain, so the answer is no.") == "yes", "the FIRST 'the answer is', to the end of ITS line"
    assert ex("so the answer is yes. So the answer is no.") == "yes. So the answer is no", "the rule is line based (lm-evaluation-harness: the rest of the line)"
    assert ex("nothing here") is None and ex("So the answer is \nmore") is None and ex("") is None
    assert ex("So the answer is 3..") == "3.", "ONE trailing period only"


def test_bbh_metric_is_exact_and_case_sensitive():
    doc = Doc(task_name="t", query="q", choices=["(B)"], gold_index=0)
    m = bbh_base.BBH_BASE_METRIC
    assert _score(m, doc, "... So the answer is (B).") == {"em": 1.0, "answer_found": 1.0}
    assert _score(m, doc, "... So the answer is (C).") == {"em": 0.0, "answer_found": 1.0}
    assert _score(m, doc, "... So the answer is B.") == {"em": 0.0, "answer_found": 1.0}
    assert _score(m, doc, "no answer line") == {"em": 0.0, "answer_found": 0.0}
    yes = Doc(task_name="t", query="q", choices=["yes"], gold_index=0)
    assert _score(m, yes, "So the answer is Yes.")["em"] == 0.0 and _score(m, yes, "So the answer is yes.")["em"] == 1.0


def test_bbh_prompt_layout():
    tasks = _tasks("bbh_base:cot|3")
    assert len(tasks) == 27 and all(n.startswith("bbh_base:") and n.endswith(":cot|3") for n in tasks)
    pm = PromptManager(use_chat_template=False)
    for sub, tail in (("navigate", "\n"), ("sports_understanding", "")):
        docs = _need(tasks[f"bbh_base:{sub}:cot|3"].get_docs, 2)
        text = pm.prepare_prompt(docs[0])
        d = BBH_COT_DEMOS[sub]
        assert text.startswith(d["description"] + "\n\nQ: " + d["shots"][0]["input"] + "\nA: " + d["shots"][0]["answer"] + "\n\nQ: " + d["shots"][1]["input"])
        assert text.count("\n\nQ: ") == 4 and text.endswith("\nA: Let's think step by step." + tail)
        assert docs[0].stop_sequences == ["\n\n", "\nQ:"] and docs[0].generation_size == 1024 and len(docs[0].fewshot_samples) == 3, "lm-evaluation-harness 1024 + OLMES stops, the bare Q dropped"
    assert len(_need(tasks["bbh_base:snarks:cot|3"].get_docs)) == 178


def test_bbh_exemplars_are_not_scored_items():
    for sub in ("boolean_expressions", "snarks"):
        task = _tasks(f"bbh_base:{sub}:cot|3")[f"bbh_base:{sub}:cot|3"]
        inputs = {d.query for d in _need(task.get_docs)}
        for shot in BBH_COT_DEMOS[sub]["shots"]:
            assert not any(shot["input"] in q for q in inputs)


# ---- mlmm_hellaswag few-shot ---------------------------------------------------------------------------------------------------

def test_mlmm_hellaswag_five_held_out_shots():
    from lighteval.tasks.multilingual.tasks import mlmm_hellaswag as mh

    names = ["mlmm_hellaswag:deu:mcf_em|5", "mlmm_hellaswag:zho:mcf_em|5", "mlmm_hellaswag:deu:mcf_em|0"]
    tasks = _tasks(",".join(names), multilingual=True)
    deu5, zho5, deu0 = (_need(tasks[n].get_docs) for n in names)
    repo, revision, split = mh._source(mh.Language.GERMAN)
    rows = _need(mh.load_rows, repo, "de", split, revision)
    assert len(deu5) == len(deu0) == len(rows) - mh.HELDOUT_SHOTS == 9363, "the same scored rows at every K"
    assert len(zho5) == 9261
    assert {len(d.fewshot_samples) for d in deu5} == {5} and {len(d.fewshot_samples) for d in zho5} == {5} and {len(d.fewshot_samples) for d in deu0} == {0}
    shots, keys = mh._heldout(mh.Language.GERMAN.value)
    assert len(shots) == 5 and len(keys) == 5 and {str(r["label"]) for r in shots} == {"0", "1", "2", "3"}, "distinct labels first"
    assert mh._heldout(mh.Language.GERMAN.value)[0] == shots, "deterministic"
    assert {mh._row_key(r) for r in rows} - keys and len({mh._row_key(r) for r in rows} - keys) >= len(deu5), "the shots are rows of the split that are not scored"
    for d in deu5[:3]:                                                                          # the same 5 rows, in the same order, in every prompt
        assert [s.id for s in d.fewshot_samples] == [f"few_shots_{i}" for i in range(5)]


# ---- EuroEval's few-shot rule for xquad (5 shots) and multi_wiki_qa (2 shots): ONE fixed set per language, the shots and their contexts held out of scoring ------------------

def _euroeval_reference(dataset, num_few_shots, random_seed=4242):
    import datasets

    datasets.disable_caching()                                                            # the copied EuroEval code filters / shuffles datasets.Dataset objects: no cache-*.arrow files in the shared cache
    """EuroEval `extract_few_shot_examples` (question-answering task group) + `_extract_question_answering_examples`, generation_utils.py @ b6288706 lines 1137-1179 and 1287-1311, copied
    (the global `random.seed` is the only thing not copied verbatim: it is the same sequence as a local Random)."""
    import random

    shuffled_train = dataset.shuffle(seed=random_seed)
    for max_num_tokens in [512, 1024, 2048, 4096, 8192]:
        train_with_short_examples = shuffled_train.filter(lambda example: len(example["context"]) < max_num_tokens)
        num_short_examples = len(train_with_short_examples)
        if num_short_examples >= num_few_shots:
            break
    else:
        raise ValueError("Could not find enough short examples for few-shot learning.")
    shuffled_train = train_with_short_examples.shuffle(seed=random_seed)
    few_shot_examples = list()
    while len(few_shot_examples) < num_few_shots and len(shuffled_train) > 0:
        example = shuffled_train.select(range(1))[0]
        few_shot_examples.append(example)
        shuffled_train = shuffled_train.filter(lambda x: x["context"] != example["context"])
    random.seed(random_seed)
    random.shuffle(few_shot_examples)
    return few_shot_examples


def test_euroeval_shots_on_a_small_dataset_escalate_the_length_limit_and_keep_contexts_distinct():
    import datasets
    from datasets import Dataset

    datasets.disable_caching()

    from lighteval.tasks.tasks import euroeval_shots as E

    rows = [{"context": "a" * 600, "question": f"q{i}", "answers": {"text": ["x"], "answer_start": [0]}} for i in range(5)]       # one context of 600 characters, 5 questions
    rows += [{"context": "b" * 700 + str(j), "question": f"r{j}", "answers": {"text": ["x"], "answer_start": [0]}} for j in range(4)]  # four more contexts, each with one question
    rows += [{"context": "c" * 50, "question": "s", "answers": {"text": [""], "answer_start": [0]}}]                               # a short context, but NO answer
    ds = Dataset.from_list(rows)                                                           # in memory, no cache files
    shots = E.select_shots(ds, 3, valid=E.has_answer)
    assert len({s["context"] for s in shots}) == 3 and all(len(s["context"]) < 1024 for s in shots), "no context under 512 characters has an answer: the limit escalates to 1024"
    assert [s["question"] for s in shots] == [s["question"] for s in E.select_shots(ds, 3, valid=E.has_answer)], "deterministic"
    assert [s["question"] for s in shots] == [s["question"] for s in _euroeval_reference(ds.filter(E.has_answer), 3)], "= EuroEval's own code on the same rows"
    assert all("c" * 50 != s["context"] for s in shots), "a row without a non-empty answer is not a candidate"
    try:
        E.select_shots(ds.filter(E.has_answer), 20)
        raise AssertionError("fewer short examples than shots at every limit must raise (EuroEval: InvalidBenchmark)")
    except ValueError as e:
        assert "enough short examples" in str(e)


def test_euroeval_shots_xquad_and_multi_wiki_qa_are_one_fixed_set_and_held_out():
    import datasets

    datasets.disable_caching()                                                            # the reference code and `rows.filter` below run on the cached dataset: no cache-*.arrow files
    from lighteval.tasks.multilingual.tasks import multi_wiki_qa, xquad
    from lighteval.tasks.templates.qa import get_qa_prompt_function
    from lighteval.tasks.tasks import euroeval_shots as E
    from lighteval.utils.language import Language

    assert E.SEED == 4242 and E.THRESHOLDS == (512, 1024, 2048, 4096, 8192) and xquad.HELDOUT_SHOTS == 5 and multi_wiki_qa.HELDOUT_SHOTS == 2
    cases = (("xquad:deu:gen", 5, "validation", get_qa_prompt_function(Language.GERMAN, xquad._adapter)),
             ("multi_wiki_qa:deu:gen", 2, "train", get_qa_prompt_function(Language.GERMAN, multi_wiki_qa._make_qa_adapter())))
    for base, k, split, plain in cases:
        spec = f"{base}|{k}"
        tasks = _tasks(f"{spec},{base}|0", multilingual=True)
        task, zero = tasks[spec], tasks[f"{base}|0"]
        docs, docs0 = _need(task.get_docs), _need(zero.get_docs)
        rows = task.dataset[split]
        shots = docs[0].fewshot_samples
        assert len(docs) == len(docs0) and all([s.id for s in d.fewshot_samples] == [s.id for s in shots] for d in docs[::13]), "ONE fixed shot set for every scored item, the same at |0 scoring"
        pick = E.select_shots(rows, k, valid=E.has_answer)
        ref = _euroeval_reference(rows.filter(E.has_answer), k)
        assert len(shots) == k and len({r["context"] for r in ref}) == k, "distinct contexts"
        assert [r["question"] for r in ref] == [r["question"] for r in pick], f"{spec}: select_shots = EuroEval's code on the pinned rows"
        assert all(s.query.startswith(r["context"]) and s.query.endswith(r["question"] + "\nAntwort:") for s, r in zip(shots, ref)), "the task shows EuroEval's shots in EuroEval's order"
        held = {r["context"] for r in ref}
        valid = rows.filter(E.has_answer)
        limit = next(l for l in E.THRESHOLDS if sum(len(r["context"]) < l for r in valid) >= k)       # EuroEval's escalation: 512 for xquad deu, 1024 for multi_wiki_qa deu (no German article is under 512 characters)
        assert all(len(r["context"]) < limit for r in ref) and limit == (512 if base.startswith("xquad") else 1024), (spec, limit)
        n_valid = sum(E.has_answer(r) for r in rows)
        removed = sum(1 for r in rows if E.has_answer(r) and r["context"] in held)
        assert len(docs) == n_valid - removed and removed >= k, "scored = every valid item minus the shots and every item on a shot's context"
        scored_ctx = {rows[int(d.id)]["context"] for d in docs}
        assert not scored_ctx & held, "no scored item sits on a shot's context"
        for d in docs0[:20]:                                                                       # |0 prompts are the plain template's
            assert d.query == plain(dict(rows[int(d.id)]), "t").query and not d.fewshot_samples
        # K shots = the first K of the set drawn for the maximum K (|1 shows a prefix)
        one = _tasks(f"{base}|1", multilingual=True)[f"{base}|1"]
        assert [s.id for s in _need(one.get_docs, 3)[0].fewshot_samples] == [shots[0].id] and len(docs) == len(_need(one.get_docs))


if __name__ == "__main__":
    failed = skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except unittest.SkipTest as skip:
                skipped += 1
                print("skip", name, skip)
            except Exception as error:  # noqa: BLE001
                failed += 1
                import traceback

                traceback.print_exc()
                print("FAIL", name, type(error).__name__, error)
    print(f"failed: {failed}, skipped: {skipped}")
    raise SystemExit(1 if failed else 0)
