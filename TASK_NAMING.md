# Task Naming Conventions

Every task name ends with an explicit metric-type suffix. There are no bare/default names.

## Evaluation Suites

### Table 1 — BPB Prepare Suite

Tasks used to measure BPB during pre-training. All tasks run in logprob mode (`generation_size=-1`). CF tasks also produce `acc` and `acc_norm` in the same pass.

```
# Code
humaneval:bpb|3
mbpp:bpb|3
mt_mbpp:bpb|3          # expands to all 17 language subtasks

# Math
math:algebra:bpb|4
math:counting_and_probability:bpb|4
math:geometry:bpb|4
math:intermediate_algebra:bpb|4
math:number_theory:bpb|4
math:prealgebra:bpb|4
math:precalculus:bpb|4

# QA — CF (BPB merged in)
arc:easy:cf|5
arc:challenge:cf|5
mmlu:cf|5              # expands to all 57 subsets
commonsenseqa:cf|5
hellaswag:cf|5
winogrande:cf|5
siqa:cf|5
piqa:cf|5
sciq:cf|5
basic_skills:cf|5      # expands to all 6 subsets
lambada:cf             # 0-shot
med_mcqa:cf|5

# QA — standalone BPB (no fixed answer choices)
coqa:bpb               # 0-shot
drop:bpb|5
jeopardy:bpb|5
natural_questions:bpb|5
squad:bpb|5
```

---

### Table 2 — Full Evaluation Suite

#### Math (CoT generation)

```
gsm8k|8
gsm_symbolic:main|8
gsm_symbolic:p1|8
gsm_symbolic:p2|8
math:algebra:gen|4
math:counting_and_probability:gen|4
math:geometry:gen|4
math:intermediate_algebra:gen|4
math:number_theory:gen|4
math:prealgebra:gen|4
math:precalculus:gen|4
math_500
```

#### STEM QA + Non-STEM QA (MC)

```
arc:easy:mcf|5
arc:challenge:mcf|5
mmlu:mcf|5             # expands to all 57 subsets (STEM + Humanities + Social Sci + Other)
med_mcqa:mcf|5
med_qa:mcf|5
sciq:mcf|5
commonsenseqa:mcf|5
piqa:mcf|5
siqa:mcf|5
jeopardy_mc:mcf|5
```

#### GenQA / RC (completion/generation)

```
hellaswag:cf|5          # RC per-char norm
winogrande:cf|5         # RC unnormalized
lambada:cf              # RC per-char norm, 0-shot
basic_skills:cf|5       # RC per-token norm, expands to all 6 subsets
drop:gen|5
jeopardy:gen|5
natural_questions:gen|5
squad:gen|5
coqa:gen                # 0-shot
```

#### Held-out Suite

```
mmlu_pro:mcf|5
bigbench_hard|3         # expands to all 27 BBH subsets
```

---

## Suffix Reference

| Suffix    | Metric type                 | `generation_size`   | Metrics reported                | Description                                                                  |
| --------- | --------------------------- | ------------------- | ------------------------------- | ---------------------------------------------------------------------------- |
| `:mcf_em` | Greedy generation (MC only) | `1`–`5`             | `exact_match`                   | MC tasks only: generate label token, compare with EM                         |
| `:cf`     | Completion formulation      | `-1` (logprob only) | `acc`, `acc_norm`, `target_bpb` | Score full answer text via log p(choice\|context); BPB merged in for MC tasks |
| `:mcf`    | Multiple-choice formulation | `-1` (logprob only) | `acc`, `acc_norm`               | Score label tokens only (`A`, `B`, …)                                        |
| `:bpb`    | Bits-per-byte (standalone)  | `-1` (logprob only) | `target_bpb`                    | Used only where CF is not applicable (MATH, free-form GenQA)                 |
| `:gen`    | Greedy generation + F1/EM   | `50`–`1024`         | `f1`, `em` (task-specific)      | Actual text generation; answer scored with F1 or extractive match            |

**Note on BPB for multiple-choice tasks**: BPB is **merged into `:cf`** for all MC tasks — running `:cf` produces `{acc, acc_norm, target_bpb}` in one pass. There are no standalone `:bpb` tasks for ARC, MMLU, HellaSwag, etc.

## How to reference tasks in CLI

Task names follow the pattern `<task_base>:<subset>:<suffix>` (or `<task_base>:<suffix>` for single-subset tasks).

Use the prefix before the first `:` as a superset to run all subsets at once:

```
# Single task
lighteval litellm config.yaml "arc:challenge:cf|5"

# All ARC variants with CF metric (2 subsets × 1 metric)
lighteval litellm config.yaml "arc:cf|5"

# All metrics for one ARC subset
lighteval litellm config.yaml "arc:challenge|5"

# All MMLU subsets for a given metric (57 subsets)
lighteval litellm config.yaml "mmlu:cf|5"
lighteval litellm config.yaml "mmlu:mcf_em|5"
lighteval litellm config.yaml "mmlu:mcf|5"

# All metrics for one MMLU subset
lighteval litellm config.yaml "mmlu:abstract_algebra|5"

# All MMLU tasks (all subsets × all metrics)
lighteval litellm config.yaml "mmlu|5"

# All MT-MBPP language subtasks
lighteval litellm config.yaml "mt_mbpp:bpb|3"
```

## Task inventory (CLAUDE.md tasks)

### Code BPB tasks

| Task | Dataset | Subset | Eval split | ICL | Metric |
|------|---------|--------|------------|-----|--------|
| `humaneval:bpb` | `openai/openai_humaneval` | default | test | 3 | `target_bpb` |
| `mbpp:bpb` | `google-research-datasets/mbpp` | sanitized | test | 3 | `target_bpb` |
| `mt_mbpp:{lang}:bpb` (17) | `allenai/multilingual_mbpp` | `{lang}` | test | 3 | `target_bpb` |

**MT-MBPP superset**: `mt_mbpp:bpb|3` expands to all 17 language subtasks.

**17 languages**: `bash`, `c`, `cpp`, `csharp`, `go`, `haskell`, `java`, `javascript`, `matlab`, `php`, `python`, `r`, `ruby`, `rust`, `scala`, `swift`, `typescript`.

### Math

| Task | Dataset | Eval split | ICL | Metric |
|------|---------|------------|-----|--------|
| `math:{subset}:bpb` (7) | `EleutherAI/hendrycks_math` | test | 4 | `target_bpb` |
| `math:{subset}:gen` (7) | `EleutherAI/hendrycks_math` | test | 4 | `expr_gold_metric` |

Subsets: `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus`.

### MMLU

| Superset | Expands to | # tasks |
|----------|------------|---------|
| `mmlu` | all subsets × all metrics | 171 (57×3) |
| `mmlu:cf` | `mmlu:{subset}:cf` for all 57 subsets | 57 |
| `mmlu:mcf_em` | `mmlu:{subset}:mcf_em` for all 57 subsets | 57 |
| `mmlu:mcf` | `mmlu:{subset}:mcf` for all 57 subsets | 57 |
| `mmlu_redux:cf` | all redux subsets with CF | 57 |

Dataset: `lighteval/mmlu`. Each `:cf` task reports `{acc, acc_norm, target_bpb}`.

### Multiple-choice QA tasks

| Task | Dataset | Eval split | ICL | `:cf` metrics | `:mcf` metrics |
|------|---------|------------|-----|---------------|----------------|
| `arc:challenge` / `arc:easy` | `allenai/ai2_arc` | test | 5 | acc, acc_norm, bpb | acc, acc_norm |
| `commonsenseqa` | `tau/commonsense_qa` | validation | 5 | acc, acc_norm, bpb | acc, acc_norm |
| `hellaswag` | `Rowan/hellaswag` | validation | 5 | acc, acc_norm, bpb | acc, acc_norm |
| `winogrande` | `allenai/winogrande` (xl) | validation | 5 | acc, acc_norm, bpb | acc, acc_norm |
| `siqa` | `lighteval/siqa` | validation | 5 | acc, acc_norm, bpb | acc, acc_norm |
| `piqa` | `lighteval/piqa` | validation | 5 | acc, acc_norm, bpb | acc, acc_norm |
| `sciq` | `allenai/sciq` | test | 5 | acc, acc_norm, bpb | acc, acc_norm |
| `med_mcqa` | `lighteval/med_mcqa` | validation | 5 | acc, acc_norm, bpb | acc, acc_norm |
| `jeopardy_mc:cf` | `allenai/jeopardy_mc` | test | 0 | acc, acc_norm, bpb | — |
| `jeopardy_mc:mcf` | `allenai/jeopardy_mc` | test | 0 | — | acc, acc_norm |

Note: `siqa` and `piqa` use `lighteval/*` wrapper repos (same data as `allenai/social_i_qa` / `ybisk/piqa`); both sources require the script fallback in `download_dataset_worker`.

### Single-answer completion tasks (CF only)

| Task | Dataset | Eval split | ICL | Metrics |
|------|---------|------------|-----|---------|
| `lambada:cf` | `EleutherAI/lambada_openai` | test | 0 | acc_norm (char), bpb |
| `basic_skills:{subset}:cf` (6) | `allenai/basic-skills` | validation | 5 | acc_norm (token), bpb |

**Basic Skills subsets**: `arithmetic`, `string_operations`, `coding`, `logical_reasoning`, `common_knowledge`, `pattern`.

**Lambada cloze variants** (perplexity only):

| Task | Dataset | Prompt |
|------|---------|--------|
| `lambada:standard_cloze` | `cimec/lambada` | `{context} ____.  ->` |
| `lambada:openai_cloze` | `EleutherAI/lambada_openai` | `{context} ____.  ->` |

### Free-form GenQA tasks

| Task | Dataset | Eval split | ICL | `:bpb` | `:gen` gen_size | `:gen` metrics |
|------|---------|------------|-----|--------|-----------------|----------------|
| `coqa` | `EleutherAI/coqa` (parquet) | validation | 0 | `target_bpb` | 50 | f1, em |
| `drop` | `lighteval/drop_harness` | validation | 5 | `target_bpb` | 100 | f1 (DROP) |
| `jeopardy` | `soldni/jeopardy` (mosaicml_gauntlet, 2117) | train | 5 | `target_bpb` | 50 | f1, em |
| `natural_questions` | `google-research-datasets/nq_open` | validation | 5 | `target_bpb` | 50 | f1, em |
| `squad` | `allenai/squad` (v1.1) | validation | 5 | `target_bpb` | 50 | f1, em |

Prompt formats:
- **CoQA**: `Passage: {story}\n\nFinal question:\n\nQuestion: {q}\nAnswer:` — stop `["\n\n"]`
- **DROP**: `Passage: {passage}\nQuestion: {question}\nAnswer:` — stop `["\n"]`
- **Jeopardy**: `Category: {cat}\nQuestion: {q}\nAnswer:` — stop `["\n\n", "Question:", "Category:"]`
- **NaturalQs**: `Question: {question}\nAnswer:` — stop `["Question:", "Q:", "\n\n"]`
- **SQuAD**: `Title: {title}\n\nBackground: {context}\n\nQuestion: {question}\n\nAnswer:` — stop `["Title:", "\n\n"]`

### CoT generation tasks

| Task | Dataset | ICL | gen_size | Metric | Subsets |
|------|---------|-----|----------|--------|---------|
| `gsm8k` | `openai/gsm8k` | 8 | 512 | `expr_gold_metric` | — |
| `gsm_symbolic:{main,p1,p2}` | `apple/GSM-Symbolic` | 8 | 512 | `expr_gold_metric` | 3 |
| `math_500` | `HuggingFaceH4/MATH-500` | 0 | 1024 | `expr_gold_metric` | — |
| `bigbench_hard:{subset}` (27) | `lukaemon/bbh` | 3 | 1024 | em (after extraction) | 27 |

---

## Metric definitions

**`:mcf_em`** — greedy decode (temperature=0), compare output to gold with exact_match.

**`:cf`** (completion formulation) — score full candidate answer text:
```
score_i = log p(answer_i | prompt)
```
Prediction = argmax. Normalizations: per-char (`LogProbCharNorm`) → `acc_norm`. BPB also computed from gold choice logprob in the same pass.

**`:mcf`** (multiple-choice formulation) — prompt shows labeled options, score only the label token:
```
score_i = log p(" A" | prompt)   # or " B", " C", " D"
```
Prediction = argmax label score. Reports `acc` and `acc_norm`.

**`:bpb`** (bits-per-byte, standalone) — no choice ranking, gold continuation only:
```
BPB = -log2 p(gold | prompt) / bytes_utf8(gold)
```
Lower is better. Used for MATH and free-form GenQA. For MC tasks, BPB is reported inside `:cf`.

**`:gen`** (greedy generation) — autoregressively decode up to `generation_size` tokens (temperature=0):
```
y = argmax_v p(v | prompt, y_<t)
```
Scored with F1 + EM (GenQA) or `expr_gold_metric` (math/CoT).
