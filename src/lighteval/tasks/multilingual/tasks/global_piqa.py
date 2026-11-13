"""
name:
Global PIQA (non-parallel)

dataset:
mrlbenchmarks/global-piqa-nonparallel

abstract:
Global PIQA is a participatory commonsense reasoning benchmark written by hand by researchers from over 65 countries:
136 language varieties, about 100 test examples each, every example a sentence stem ("prompt") with two candidate
continuations ("solution0", "solution1") of which one is correct. The non-parallel split is the paper's primary split:
the examples are native to each language and over 50% refer to local foods, customs or traditions. License CC BY-SA 4.0;
the authors do NOT allow training on it (the data carries a canary string, see the dataset README): evaluation only.

Here as a letter-choice task (`mcf_em`, like piqa:mcf_em): the stem and the two solutions as options A / B, the model writes
the letter, exact match. The dataset has ONLY a test split (no dev / train), so the few-shot demonstrations are 8 fixed
items of the English PIQA training set (lighteval/piqa, seeded draw, gold letters alternating), shown in the same
format; the evaluated language never supplies its own shots.

languages:
136 varieties, named by their config (ISO 639-3 + ISO 15924 script, sometimes a region): deu_latn, cmn_hans, ...

tags:
multilingual, commonsense, multiple-choice, culture

paper:
https://arxiv.org/abs/2510.24081
"""

import random
from functools import lru_cache

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# Pinned dataset revisions (the snapshots in the offline cache): the benchmark, and English PIQA for the demonstrations.
_REVISION = "7394a871c9257746234b2f189bc585734862aa18"
_PIQA_REPO = "lighteval/piqa"
_PIQA_REVISION = "41782e6bf0ef7de82a2ca8a9feb1dca042837fae"

# The 136 configs of the non-parallel split (README of the dataset; one TSV each, split "test").
_CONFIGS = [
    "acm_arab", "acq_arab", "aeb_arab", "afb_arab", "als_latn", "amh_ethi",
    "apc_arab_jord", "apc_arab_leba", "apc_arab_pale", "apc_arab_syri", "arb_arab", "arq_arab",
    "ars_arab", "ary_arab", "arz_arab", "asm_beng", "azj_latn", "bam_latn",
    "bcc_arab", "bel_cyrl", "ben_beng", "ben_latn", "bgc_deva", "bho_deva",
    "bos_latn", "bra_deva", "bsk_arab", "btx_latn", "bul_cyrl", "cat_latn",
    "ces_latn", "ckb_arab", "ckm_latn", "cls_deva", "cmn_hans", "cmn_hant",
    "dan_latn", "deu_latn", "dhd_deva", "dje_latn", "ekk_latn", "ekp_latn",
    "ell_grek", "eng_latn", "eus_latn", "fao_latn", "fin_latn", "fra_latn_cana",
    "fra_latn_fran", "glg_latn", "guj_gujr", "hau_latn", "haw_latn", "heb_hebr",
    "hin_deva", "hin_latn", "hrv_latn", "hun_latn", "hye_armn", "ibo_latn",
    "idu_latn", "ind_latn", "isl_latn", "iso_latn", "ita_latn", "jav_latn",
    "jpn_jpan", "kan_knda", "kan_latn", "kat_geor", "kaz_cyrl", "kin_latn",
    "kir_cyrl", "kor_hang", "lin_latn", "lit_latn", "luo_latn", "mal_mlym",
    "mar_deva", "mkd_cyrl", "mni_beng", "mni_mtei", "nag_latn", "nld_latn",
    "nno_latn", "nob_latn", "npi_deva", "ory_orya", "pan_guru", "pcm_latn",
    "pes_arab", "plt_latn", "pol_latn", "por_latn_braz", "por_latn_port", "ron_latn",
    "rus_cyrl", "rwr_deva", "sin_latn", "sin_sinh", "slk_latn_sari", "slk_latn",
    "slv_latn_cerk", "slv_latn_prle", "slv_latn", "snd_arab", "snd_deva", "spa_latn_mexi",
    "spa_latn_peru", "spa_latn_spai", "srp_cyrl_torl", "srp_cyrl", "srp_latn_torl", "srp_latn",
    "sun_latn", "swe_latn", "swh_latn", "swv_deva", "tam_latn", "tam_taml",
    "tel_latn", "tel_telu", "tgl_latn", "tha_thai", "tur_latn", "uig_arab",
    "ukr_cyrl", "urd_arab", "urd_latn", "urh_latn", "uzn_latn", "vie_latn",
    "yor_latn", "yue_hant", "zsm_latn", "zul_latn",
]

_INSTRUCTION = "Given the following situation, which option is more likely to be correct? Answer with A or B.\n\n"


def global_piqa_mcf_prompt(line, task_name: str = None):
    """Letter format. The wording follows the dataset's own generation template (English instruction for every language,
    "Situation: <prompt> ...", options A / B); the model answers with the letter."""
    query = f"{_INSTRUCTION}Situation: {line['prompt']} ...\n A. {line['solution0']}\n B. {line['solution1']}\nAnswer:"
    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B"],
        gold_index=int(line["label"]),
        specific={"language": line.get("language")},
    )


@lru_cache(maxsize=1)
def _english_piqa_demos(n: int = 8) -> list[dict]:
    """n fixed demonstrations in the Global PIQA row format, from the English PIQA TRAIN split: a seeded draw with the gold
    letters alternating (B, A, B, A, ...). Called once per task by the few-shot sampler."""
    from lighteval.tasks.lighteval_task import load_dataset_pinned

    train = load_dataset_pinned(_PIQA_REPO, "plain_text", split="train", revision=_PIQA_REVISION)
    rng = random.Random(1234)
    by_label = {0: [], 1: []}
    for i in rng.sample(range(len(train)), 200):
        by_label[int(train[i]["label"])].append(train[i])
    rows = []
    for k in range(n):
        r = by_label[(k + 1) % 2].pop(0)
        rows.append({"prompt": r["goal"], "solution0": r["sol1"], "solution1": r["sol2"], "label": int(r["label"])})
    return rows


TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"global_piqa:{cfg}:mcf_em",
        prompt_function=global_piqa_mcf_prompt,
        hf_repo="mrlbenchmarks/global-piqa-nonparallel",
        hf_revision=_REVISION,
        hf_subset=cfg,
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_rows=_english_piqa_demos,
        few_shots_select="sequential",
        generation_size=1,
        stop_sequence=["\n"],
        metrics=[Metrics.exact_match],
    )
    for cfg in _CONFIGS
]
