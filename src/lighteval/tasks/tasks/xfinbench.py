"""
name:
XFinBench Cross-lingual Finance QA

dataset:
titaneval_local (local parquet - data/titaneval/xfinbench.parquet)

abstract:
Finance multiple-choice benchmark (English). The local file holds the 592 multiple-choice rows of the TEST split
(`test_set.csv`, task == "mcq") of Zhihan/XFinBench @ 9eb2a9d9, byte-identical to the TitanEval-MCQ copy (the `figure` column of
this multimodal benchmark was dropped there). Version 1 (owner 2026-11-13): 468 of the 592 rows are scored; 4 rows with the
placeholder choice list ['nan'] (numeric questions about "the attached figure") and the 120 rows that depend on a figure which is
not in the prompt are dropped (`drop_reason`, below).

Figure rule: the upstream `figure` column is non-empty for 124 of the 592 rows (4 of them are the placeholder rows, 120 are kept
by the validity rule); the local file has no such column, so the 120 rows are named by a content key (`FIGURE_KEYS`, see
`figure_key`). agents/reward_checks/domain_fixes_checks.py (titan_eval; `--upstream-csv test_set.csv --print-figure-keys`) derives the
keys from the pinned upstream file and checks them against FIGURE_KEYS; the unit test checks them against the local file (120 keys,
each matching exactly one row). The
list of dropped rows (upstream id, capability, figure file, question start) is docs/evals/tasks/_raw/xfinbench_mcf_em-dropped.tsv
in titan_eval. A text rule would give 124 (every FF-capability row + every row that says "attached", the first proposal): it keeps 5
figure items and drops 9 FF rows that need no figure, so the field is used.

Few-shot pool = the scored rows (the same filtered split, `random_sampling`, the evaluated item is excluded); 5 shots in the suite.

languages:
english

tags:
finance, multiple-choice, qa
"""

from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.normalizations import LogProbCharNorm
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.tasks.tasks.domain_drops import KeepRows
from lighteval.tasks.tasks.heldout_shots import content_key

_CF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
    Metrics.target_bits_per_byte,
]
_MCF_METRICS = [
    LogLikelihoodAccMetric(),
    LogLikelihoodAccMetric(normalization=LogProbCharNorm()),
]


def xfinbench_cf_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\nAnswer:",
        choices=[" " + c for c in line["choices"]],
        gold_index=line["answer_index"],
    )


def xfinbench_mcf_prompt(line, task_name: str = None):
    choices = line["choices"]
    labels = list("ABCDEFGHIJ"[: len(choices)])
    options = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    return Doc(
        task_name=task_name,
        query=f"Question: {line['question']}\n{options}\nAnswer:",
        choices=[f" {l}" for l in labels],
        gold_index=line["answer_index"],
    )


def _valid_row(row) -> bool:
    """Skip rows with empty choices (xfinbench has a few invalid rows)."""
    choices = row.get("choices") or []
    ans = row.get("answer_index")
    return bool(choices) and any(c.strip() for c in choices) and ans is not None and 0 <= ans < len(choices)


def figure_key(row) -> str:
    """Content key of a row (question, options, gold index): names a row without its position in the file."""
    return content_key(row["question"], "\x1e".join(row["choices"]), int(row["answer_index"]))


# Rows whose upstream `figure` field is set (the question needs a chart / diagram / tree that is not in the prompt): 120 keys.
FIGURE_KEYS = frozenset({
    "071620fa76f05251", "089174ee041ff7b5", "0e91e7a48220f297", "0ed818116bcee8e7",
    "12f743f86f86b5f8", "13a9f226a4fe9d28", "146dbde2b9bb7cef", "1485ac307f989ea4",
    "15fbb890e5aeca82", "1682ba30ae188dfd", "18d17d70597d69ad", "1b8a8aec14df8115",
    "1dac61a9bf285f59", "1e20315d5188c99c", "21bcbef1056c819d", "226e77cde4c911d9",
    "230d5775f8be8037", "284777f1b67c0046", "2c81057e02686511", "2f8d21cc7f033597",
    "2fd60099e30a937c", "300baa3d06da758a", "32689536d4f2c2d7", "35aa9457589e4c1b",
    "39547df15ec241de", "4127fe0d4b0294e9", "42ceaf0321d8ce72", "4431cddb7d4a1626",
    "471951e1d04d5c9d", "4a80bd461195046c", "4acf73a74d8a9a32", "4e21daa6e97acc34",
    "4ffcb3b320d9b98d", "5359dfa4d5f8ed30", "54ceb1e0553bd2d9", "55e3a65dedb218c4",
    "5656ef6f8c767665", "56f4a15181aec91f", "583750346cd2db6d", "58b6ef727d14a7b8",
    "59fb1c9ce712e840", "5a5a8a6adc84453f", "5d6cfad8e7956d9c", "5ff86d73f543b43f",
    "61b5c87be7b6573d", "6398b380185174c9", "6511253b42a8e2ad", "655ce96d988326b2",
    "65893d5cfc4c000e", "66dae17c03f02439", "6802d2222527b853", "6b70cbaeeac28422",
    "6c7d83b478f4767c", "6d1b305ea879a48e", "6ee1c4452c5edd78", "6efda105bd6b9701",
    "75166cbc8c485ce5", "75d18f78868e157e", "77613d2083357618", "77a77dbf306ef093",
    "77cd947e7be9e9fd", "78423f37f34d6234", "784617c36ac0879d", "7acc3b265fc76898",
    "7cc89ddb23ca6f24", "7ccdda1376871733", "7e0e9f4ec18f78a6", "7f0467dd82691463",
    "8685280df1be26bc", "8a4af46e7191d940", "8b6d1d25f398c30a", "8b955845d36a5c4d",
    "8d00110290b0c187", "8ee7e7e76735fff3", "8f2974289248544c", "92d285f84c4e3592",
    "94fce5532bbe8cd6", "950ea7902a93d5ff", "9526cca8729ab0b3", "964c2a95869e99ea",
    "9e71fd77f765ecd0", "a07d72ec35837c1e", "a0fae448615a5622", "a49d0fdf5f79446f",
    "a5cefae31c2ac4d7", "a84a90b9da91092e", "a982f12a13f8838a", "aecf48a244078088",
    "b0712a1548a425a8", "b9c26038d1081853", "bce24f0e9996ae1c", "be4d114fe673f95d",
    "bebae2a0bdc3d698", "bebcd487882e5903", "bfa3ef27c209ee18", "c06bdde0eab5e758",
    "c08645a510ee09aa", "c11e2aadcc4e6176", "c7b1dcf676e47a96", "d255a093bfd0742d",
    "d7f1572e517e460d", "d81cf227a37057f6", "db02b2521a87ee8d", "df949271a7181090",
    "e0a7e95e3520c0b0", "e1792455b80cfeb4", "e2fde46634612427", "e377fecf2c842744",
    "e3acfa67736838f0", "e661f844b1022deb", "e75ff05ed0c93028", "e839f3d0ca299438",
    "eaae884794521ff5", "eb52b0ba24ec3c30", "f038b7d9a216a8bb", "f5076ca86276649f",
    "fbdccbce0b0f35f5", "fdfcbffbbd5d8915", "fecc55f825d7ed4a", "ffc268c4039d5a5e",
})
assert len(FIGURE_KEYS) == 120


def drop_reason(row):
    """None = scored; "invalid" (placeholder choice list) or "figure" (the question depends on a figure that is not shown)."""
    if not _valid_row(row):
        return "invalid"
    if figure_key(row) in FIGURE_KEYS:
        return "figure"
    return None


# The digest of FIGURE_KEYS is part of the datasets cache key (see domain_drops); bump `version` and `rule` with every change of the rule.
_KEEP = KeepRows("xfinbench-v1: valid choices and no upstream figure (120 keys)", drop_reason, parts=sorted(FIGURE_KEYS))


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"xfinbench:{suffix}",
        prompt_function=fn,
        hf_repo="titaneval_local",
        hf_subset="xfinbench",
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
        ("cf",     xfinbench_cf_prompt,  _CF_METRICS,          -1),
        ("mcf",    xfinbench_mcf_prompt, _MCF_METRICS,         -1),
        ("mcf_em", xfinbench_mcf_prompt, [Metrics.exact_match],  1),
    ]
]
