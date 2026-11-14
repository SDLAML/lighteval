# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
name:
Judge-graded chat evals (titan_eval, post-trained models): SimpleQA, AA-Omniscience (public set), AA-LCR, Humanity's Last Exam (text only)

dataset:
lighteval/SimpleQA, ArtificialAnalysis/AA-Omniscience-Public, ArtificialAnalysis/AA-LCR, cais/hle

abstract:
Generation half of four benchmarks that need a judge model. Compute nodes are offline, so the judging is NOT done inside lighteval: these tasks only produce the answers (prompt as
in the benchmark's own harness, the model's chat template and thinking switch from the run's model settings, scored text = what follows the LAST </think>; the lighteval metric is
`answered`, the share of non-empty answers) and put into `Doc.specific` everything the judge stage needs: `{"question": the question as the grader prompt shows it, "gold": the
reference answer, ...metadata}`. titan_eval/evals/judge.py then reads the answers back from the details, grades them with the benchmark's OFFICIAL grader prompt through a local judge
model (titan_eval/evals/judge_graders.py) and adds the benchmark's metrics to scores.json. Tasks are named `<name>_chat`.
  simpleqa_chat       question only, as simple-evals (one user message, no system prompt); 4,321 test questions (+ 5 few-shot rows, not scored)
  aa_omniscience_chat the dataset card's OMNISCIENCE_ANSWER_PROMPT (domain, subtopic) + the question in ONE user message (as the fork's aa_omniscience); 600 questions
  aa_lcr_chat         the dataset card's prompt template: the documents of the question's set (in the order of data_source_filenames) + the question; 100 questions with 72k to 115k tokens
                      (cl100k_base) of input: needs vllm.max_model_len >= ~160k (the task yaml sets it); the documents come from the dataset's extracted_text zip
  hle_chat            the question as the user message; the repo's SYSTEM_PROMPT (Explanation / Answer / Confidence) is the run's system prompt (the task yaml's chat.system_prompt);
                      text-only questions (no image) of the 2,500; the dataset is GATED: the fetch needs the owner's accepted terms

languages:
english

tags:
chat, factuality, long-context, judge

paper:
https://arxiv.org/abs/2411.04368 (SimpleQA), https://arxiv.org/abs/2511.13029 (AA-Omniscience), https://arxiv.org/abs/2501.14249 (HLE)
"""

import threading
import unicodedata
import zipfile

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.tasks.tasks.chat_evals import BUDGET


# Pinned revisions (commit hashes, the same ones titan_eval's configs/datasets.lock.yaml lists)
SIMPLEQA_REPO, SIMPLEQA_REVISION = "lighteval/SimpleQA", "d785eed336ea739b1c01683a6edc977a47e4cb8b"
OMNISCIENCE_REPO, OMNISCIENCE_REVISION = "ArtificialAnalysis/AA-Omniscience-Public", "e4883edbb9f5ccf2b2a8fdc6fb65e01a58e99849"
LCR_REPO, LCR_REVISION = "ArtificialAnalysis/AA-LCR", "9a77ef56b717057ade24ceab4d273712a0b4f19e"       # v1.1 (the current one); v1.0.0 = bdae010bbce259820c0e34c1d7cce210d966fb75
LCR_ZIP = "extracted_text/AA-LCR_extracted-text.zip"
HLE_REPO, HLE_REVISION = "cais/hle", "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"                       # gated: not fetched yet (the revision is the one the hub lists)

# files of a dataset repo that a task reads next to the rows (`titan-eval prefetch` downloads them too: titan_eval/evals/prefetch.py EXTRA_FILES hook)
EXTRA_FILES = {"aa_lcr_chat": [(LCR_REPO, LCR_ZIP, LCR_REVISION)]}


# ------------------------------------------------------------------------------------------------------- the lighteval metric
class _Answered(SampleLevelComputation):
    """`answered`: the share of the prompt's samples whose scored text (after the last </think>) is not empty. The only lighteval-side metric of a judge-graded task; the
    benchmark's metrics are computed by the judge stage of titan_eval (a judged run puts them into scores.json next to this one)."""

    def compute(self, doc, model_response, **kwargs) -> dict[str, float]:
        texts = list(model_response.final_text)
        return {"answered": float(np.mean([1.0 if t and t.strip() else 0.0 for t in texts])) if texts else 0.0}


answered_metric = SampleLevelMetricGrouping(
    metric_name=["answered"],
    sample_level_fn=_Answered(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn={"answered": np.mean},
    higher_is_better={"answered": True},
)


def _doc(task_name, query, question, gold, **meta) -> Doc:
    return Doc(
        task_name=task_name,
        query=query,
        choices=[str(gold)],
        gold_index=0,
        specific={"question": question, "gold": str(gold), **meta},
    )


# ------------------------------------------------------------------------------------------------------- SimpleQA
def simpleqa_chat_prompt(line, task_name: str = None):
    """One user message = the question, nothing else (simple-evals `sampler([{"role": "user", "content": problem}])`)."""
    return _doc(task_name, line["problem"], line["problem"], line["answer"])


# ------------------------------------------------------------------------------------------------------- AA-Omniscience
# the dataset card's OMNISCIENCE_ANSWER_PROMPT, verbatim (titan_eval/evals/judge_graders.py carries the same text and its sha256; the fork's aa_omniscience.py has `{topic}` where the
# card says `{subtopic}`, which is the column of the data)
OMNISCIENCE_ANSWER_PROMPT = (
    "You are answering questions about {domain}, and in particular {subtopic}.\n"
    "You will be given a question, answer with JUST the answer (no explanation).\n"
    "If you do not know the answer, or you need more context or tools to answer the question,\n"
    "be clear about this - it is better that you say this than get the wrong answer."
)


def aa_omniscience_chat_prompt(line, task_name: str = None):
    query = OMNISCIENCE_ANSWER_PROMPT.format(domain=line["domain"], subtopic=line["subtopic"]) + "\n\n" + line["question"]
    return _doc(task_name, query, line["question"], line["answer"], domain=line["domain"], topic=line["topic"], subtopic=line["subtopic"], question_id=int(line["question_id"]))


# ------------------------------------------------------------------------------------------------------- AA-LCR
# the dataset card's prompt template (README of ArtificialAnalysis/AA-LCR, "Prompt Template"); documents are joined exactly as its code does
LCR_PROMPT = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

_zip_lock = threading.Lock()
_zip_cache: dict = {}


def _lcr_zip_index():
    """{NFC file name inside the zip: ZipInfo} of the dataset's extracted-text zip, read from the hub cache (offline-safe: the file is the pinned revision's). The zip stores the names without the
    UTF-8 flag, i.e. as cp437 text of UTF-8 bytes, and in NFD (`Başev` as `s` + combining cedilla): both undone for the lookup."""
    with _zip_lock:
        if "index" not in _zip_cache:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(LCR_REPO, LCR_ZIP, repo_type="dataset", revision=LCR_REVISION)
            zf = zipfile.ZipFile(path)
            index = {}
            for info in zf.infolist():
                name = info.filename
                try:
                    name = name.encode("cp437").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
                index[unicodedata.normalize("NFC", name)] = info
            _zip_cache["zip"], _zip_cache["index"] = zf, index
        return _zip_cache["zip"], _zip_cache["index"]


def lcr_documents(category: str, set_id: str, filenames: list[str]) -> list[str]:
    """The texts of a question's documents in the order of `filenames` (data_source_filenames). A missing document is an error (a silently shorter context would change the task)."""
    zf, index = _lcr_zip_index()
    out = []
    for fn in filenames:
        key = unicodedata.normalize("NFC", f"lcr/{category}/{set_id}/{fn}")
        if key not in index:
            raise FileNotFoundError(f"AA-LCR document {key!r} is not in {LCR_ZIP} @ {LCR_REVISION[:8]}")
        with _zip_lock:
            out.append(zf.read(index[key]).decode("utf-8"))
    return out


def lcr_prompt(question: str, docs: list[str]) -> str:
    documents_text = "\n\n".join(f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs))
    return LCR_PROMPT.format(documents_text=documents_text, question=question)


def aa_lcr_chat_prompt(line, task_name: str = None):
    docs = lcr_documents(line["document_category"], line["document_set_id"], line["data_source_filenames"].split(";"))
    return _doc(task_name, lcr_prompt(line["question"], docs), line["question"], line["answer"], document_category=line["document_category"],
                document_set_id=line["document_set_id"], question_id=int(line["question_id"]), input_tokens_cl100k=int(line["input_tokens"]))


# ------------------------------------------------------------------------------------------------------- HLE (text only)
def hle_chat_prompt(line, task_name: str = None):
    """The question as the user message (the repo's format_message without an image). Questions with an image are dropped (text-only evaluation, as the fork's `hle`); the
    repo's SYSTEM_PROMPT is the system message (run setting: chat.system_prompt in the task yaml)."""
    if line["image"] not in (None, ""):
        return None
    return _doc(task_name, line["question"], line["question"], line["answer"], id=line["id"], answer_type=line["answer_type"], category=line["category"],
                raw_subject=line["raw_subject"])


def _cfg(name, prompt_fn, repo, revision, split, subset="default"):
    return LightevalTaskConfig(
        name=name,
        prompt_function=prompt_fn,
        hf_repo=repo,
        hf_subset=subset,
        hf_revision=revision,
        hf_avail_splits=[split],
        evaluation_splits=[split],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=BUDGET,
        stop_sequence=[],
        metrics=[answered_metric],
        version=0,
    )


TASKS_TABLE = [
    _cfg("simpleqa_chat", simpleqa_chat_prompt, SIMPLEQA_REPO, SIMPLEQA_REVISION, "test"),
    _cfg("aa_omniscience_chat", aa_omniscience_chat_prompt, OMNISCIENCE_REPO, OMNISCIENCE_REVISION, "train"),
    _cfg("aa_lcr_chat", aa_lcr_chat_prompt, LCR_REPO, LCR_REVISION, "test"),
    _cfg("hle_chat", hle_chat_prompt, HLE_REPO, HLE_REVISION, "test"),
]
