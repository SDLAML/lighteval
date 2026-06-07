# Task Naming Conventions

Every task name ends with an explicit metric-type suffix and a shot count: `{task}:{suffix}|{n_shot}`.

## Task Overview

**48 unique English tasks** (`:cf`/`:mcf`/`:gen` variants of the same task count as one):

| Category | # | Task names |
|---|---|---|
| Code BPB (§3) | 3 | `humaneval`, `mbpp`, `mt_mbpp` |
| Math (§3+§4) | 2 | `math` (BPB+gen), `math_500` |
| CoT Reasoning (§4) | 5 | `gsm8k`, `gsm_plus`, `gsm_symbolic`, `bigbench_hard`, `agieval_eng_em` |
| English MC QA (§5) | 31 | `mmlu`, `mmlu_pro`, `arc`, `commonsenseqa`, `siqa`, `piqa`, `sciq`, `hellaswag`, `winogrande`, `swag`, `openbookqa`, `qasc`, `boolq`, `med_mcqa`, `med_qa`, `pubmedqa`, `headqa`, `gpqa`, `jeopardy_mc`, `truthfulqa`, `cybermetric`, `secqa`, `mascqa`, `formationeval`, `teleqna`, `labbench`, `preflight`, `chembench`, `esgenius`, `xfinbench`, `geobench` |
| English GenQA (§6) | 10 | `coqa`, `drop`, `jeopardy`, `natural_questions`, `squad`, `squad_v2`, `triviaqa`, `popqa`, `wikifact`, `simpleqa` |
| Lambada & Basic Skills (§7) | 2 | `lambada`, `basic_skills` |
| **Total** | **53** | |

---

---

## 1. Suffix Reference

| Suffix | `generation_size` | Metrics reported | Description |
|--------|-------------------|-----------------|-------------|
| `:cf` | `-1` | `acc`, `acc_norm`, `target_bpb`* | Logprob on full answer texts; BPB merged in |
| `:mcf` | `-1` | `acc`, `acc_norm` | Logprob on label tokens only (`A`, `B`, …) |
| `:mcf_em` | `1` | `em` | Greedy-decode the label token, exact match |
| `:gen` | task-specific | `f1`, `em` (normalized) | Generate free text; scored with normalized F1 + EM |
| `:bpb` | `-1` | `target_bpb` | Standalone BPB; used for code, math, or decoupled from `:gen` / `:cf` |

\*BPB merged into `:cf` applies to **English MC QA tasks (§5) only**. For lambada and basic_skills (§7), BPB uses a different prompt and is a separate `:bpb` config — `:cf` for those tasks does **not** include BPB.

**Gen tasks**: `:gen` and `:bpb` are separate configs (different prompts). EM and F1 for `:gen` use
`harness_triviaqa_normalizer` (lowercase + remove punctuation) on both gold and prediction.
Exception: `drop:gen` uses `Metrics.drop` (span/number/date-aware normalization).

---

## 2. How to Run (CLI)

```bash
# Single task
lighteval litellm config.yaml "arc:challenge:cf|5"

# All variants of one task
lighteval litellm config.yaml "arc:challenge|5"

# All subsets of one task + one metric
lighteval litellm config.yaml "mmlu:cf|5"            # 57 subsets
lighteval litellm config.yaml "wikifact:gen|5"       # 81 relation subsets

# All subsets × all metrics
lighteval litellm config.yaml "mmlu|5"               # 57 × 3 = 171 tasks
lighteval litellm config.yaml "arc|5"                # 2 × 3 = 6 tasks

# Multilingual (requires --load-multilingual flag in runner)
lighteval litellm config.yaml "global_mmlu:cf|5"
lighteval litellm config.yaml "mlmm_arc:deu:mcf|5"
```

---

## 3. Code & Math BPB Tasks

BPB over the gold continuation only (`generation_size=-1`). No accuracy metric.

| Task | Dataset | Eval | FS | ICL |
|---|---|---|---|---|
| `humaneval:{lang}:bpb` | `openai/openai_humaneval` | test | — | 3 |
| `mbpp:bpb` | `google-research-datasets/mbpp` (sanitized) | test | — | 3 |
| `mt_mbpp:{lang}:bpb` (17) | `allenai/multilingual_mbpp` | test | — | 3 |
| `math:{subset}:bpb` (7) | `EleutherAI/hendrycks_math` | test | — | 4 |

**MT-MBPP languages (17):** `bash`, `c`, `cpp`, `csharp`, `go`, `haskell`, `java`, `javascript`,
`matlab`, `php`, `python`, `r`, `ruby`, `rust`, `scala`, `swift`, `typescript`.

**Math subsets (7):** `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`,
`number_theory`, `prealgebra`, `precalculus`.

```bash
humaneval:bpb|3
mbpp:bpb|3
mt_mbpp:bpb|3              # all 17 languages
mt_mbpp:python:bpb|3       # single language
math:bpb|4                 # all 7 subsets
math:algebra:bpb|4         # single subset
```

---

## 4. Math & CoT Reasoning Tasks

All tasks in this section generate free text and score with extractive match metrics.

| Task | Dataset | Few-shot split | Rec. ICL | gen_size | Metric |
|---|---|---|---|---|---|
| `math:{subset}:gen` (7) | `EleutherAI/hendrycks_math` | `train` | 4 | 1024 | `expr_gold_metric` |
| `math_500` | `HuggingFaceH4/MATH-500` | `test`¹ | 4 | 1024 | `expr_gold_metric` |
| `gsm8k` | `openai/gsm8k` | `train` | 8 | 512 | `expr_gold_metric` |
| `gsm_plus` | `qintongli/GSM-Plus` | `testmini` | 8 | 512 | `expr_gold_metric` |
| `gsm_symbolic:{main,p1,p2}` | `apple/GSM-Symbolic` | `test`¹ | 8 | 512 | `expr_gold_metric` |
| `bigbench_hard:{subset}` (27) | `lukaemon/bbh` | `train` | 3 | 1024 | `bbh_cot_exact_match` |
| `agieval_eng_em:{subset}` (7) | `lighteval/agi_eval_en` | `dev` | 0 | 512 | `gpqa_instruct_metric` |

¹ Test-only datasets (no train split): `math_500` (`HuggingFaceH4/MATH-500`), `gsm_symbolic` (`apple/GSM-Symbolic`). Few-shot examples are drawn from the test pool via random sampling (potential leakage). For leakage-free math few-shot, prefer `math:gen|4` (draws from `hendrycks_math` train) and `gsm8k|8` (draws from GSM8K train).

> **4k context note:** `gsm8k|8` ≈ 3–4k tokens of context (borderline); `math:gen|4` ≈ 4k+ (too long). Use `gsm8k|4` and `math:gen|1` for 4k-ctx models.

**`expr_gold_metric`** — extracts mathematical expressions / LaTeX (including `\boxed{}`) from model output; scores with symbolic equivalence.

**`bbh_cot_exact_match`** — extracts text after "the answer is" from CoT output; exact match.

**`gpqa_instruct_metric`** (AGIEval) — extracts letter choice (A–E) from CoT output.

```bash
math:gen|4                 # all 7 subsets, 4-shot from train
math:algebra:gen|4         # single subset
math_500|4                 # 4-shot drawn from test pool (see ¹)
gsm8k|8                    # 8-shot from train (standard)
gsm_plus|8                 # 8-shot from testmini
gsm_symbolic:main|8
gsm_symbolic:p1|8
gsm_symbolic:p2|8
bigbench_hard|3            # all 27 subsets
bigbench_hard:boolean_expressions|3
agieval_eng_em|0           # all 7 subsets
agieval_eng_em:aqua_rat|0  # single subset
```

**Math subsets (7):** `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`,
`number_theory`, `prealgebra`, `precalculus`.

**AGIEval (English) subsets (7):** `aqua_rat`, `logiqa-en`, `lsat-ar`, `lsat-lr`, `lsat-rc`, `sat-en`, `sat-math`.

---

## 5. English MC QA Tasks

All tasks in this section expose three variants:
- `:cf|N` → `acc`, `acc_norm`, `target_bpb`
- `:mcf|N` → `acc`, `acc_norm`
- `:mcf_em|N` → `em`

Exceptions are noted per task.

### MMLU

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `lighteval/mmlu` | test | dev | 5 |

57 subjects across STEM, Humanities, Social Sciences, Other.

```bash
mmlu:cf|5                         # all 57 subsets
mmlu:mcf|5
mmlu:mcf_em|5
mmlu:abstract_algebra:cf|5        # single subject
mmlu|5                            # all 57 × 3 variants
```

### MMLU-Pro

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `TIGER-Lab/MMLU-Pro` | test | validation | 5 | up to 10 options |

```bash
mmlu_pro:cf|5
mmlu_pro:mcf|5
mmlu_pro:mcf_em|5
mmlu_pro:cot|5                    # chain-of-thought + extractive match (separate config)
```

### ARC

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `allenai/ai2_arc` | test | train | 5 |

```bash
arc:cf|5                          # both easy + challenge
arc:mcf|5
arc:mcf_em|5
arc:easy:cf|5                     # single subset
arc:challenge:mcf_em|5
```

### CommonsenseQA

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `tau/commonsense_qa` | validation | train | 5 |

```bash
commonsenseqa:cf|5
commonsenseqa:mcf|5
commonsenseqa:mcf_em|5
```

### SIQA

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `lighteval/siqa` | validation | train | 5 |

```bash
siqa:cf|5
siqa:mcf|5
siqa:mcf_em|5
```

### PIQA

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `lighteval/piqa` | validation | train | 5 |

```bash
piqa:cf|5
piqa:mcf|5
piqa:mcf_em|5
```

### SciQ

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `allenai/sciq` | test | train | 5 |

```bash
sciq:cf|5
sciq:mcf|5
sciq:mcf_em|5
```

### HellaSwag

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `Rowan/hellaswag` | validation | train | 5 | sentence completion |

```bash
hellaswag:cf|5
hellaswag:mcf|5
hellaswag:mcf_em|5
```

### WinoGrande

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `allenai/winogrande` (xl) | validation | train | 5 | cloze / pronoun resolution |

```bash
winogrande:cf|5
winogrande:mcf|5
winogrande:mcf_em|5
winogrande:bpb|5             # OLMO-style partial evaluation BPB (separate config)
```

### SWAG

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `allenai/swag` (regular) | validation | train | 5 |

```bash
swag:cf|5
swag:mcf|5
swag:mcf_em|5
```

### OpenBookQA

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `allenai/openbookqa` (main) | test | train | 5 |

```bash
openbookqa:cf|5
openbookqa:mcf|5
openbookqa:mcf_em|5
```

### QASC

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `allenai/qasc` | validation | train | 5 | 8-choice, requires two facts |

```bash
qasc:cf|5
qasc:mcf|5
qasc:mcf_em|5
```

### BoolQ

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `lighteval/boolq_helm` | validation | train | 5 | binary yes/no |

```bash
boolq:cf|5
boolq:mcf|5
boolq:mcf_em|5
```

### MedMCQA

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `lighteval/med_mcqa` | validation | train | 5 |

```bash
med_mcqa:cf|5
med_mcqa:mcf|5
med_mcqa:mcf_em|5
```

### MedQA (USMLE)

| Dataset | Eval | FS | ICL |
|---|---|---|---|
| `bigbio/med_qa` (med_qa_en_source) | test | train | 5 |

```bash
med_qa:cf|5
med_qa:mcf|5
med_qa:mcf_em|5
```

### PubMedQA

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `qiaojin/PubMedQA` (pqa_labeled) | train | train | 5 | 3-choice: yes/no/maybe |

```bash
pubmedqa:cf|5
pubmedqa:mcf|5
pubmedqa:mcf_em|5
```

### HeadQA

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `lighteval/headqa_harness` | test | train | 5 | en + es subsets |

```bash
headqa:en:cf|5
headqa:es:cf|5
headqa:cf|5                  # both subsets
headqa:mcf|5
headqa:mcf_em|5
```

### GPQA (Diamond)

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `Idavidrein/gpqa` (gpqa_diamond) | train | train | 0 | gated; choices shuffled by question hash |

```bash
gpqa:diamond:cf|0
gpqa:diamond:mcf|0
gpqa:diamond:mcf_em|0
gpqa:diamond|0               # all 3 variants
```

Note: `gpqa:diamond` (instruct CoT, `gpqa_instruct_pass_at_k`) and `gpqa:main` / `gpqa:extended`
(instruct reasoning) are separate configs for instruction-tuned evaluation.

### Jeopardy MC

OLMo Gen2MC — dedicated MC dataset derived from Jeopardy. For the generative form see `jeopardy` in §6.

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `allenai/jeopardy_mc` | test | — | 0 | CF + MCF only (no mcf_em) |

```bash
jeopardy_mc:cf|0
jeopardy_mc:mcf|0
```

### TruthfulQA (MC2)

| Dataset | Eval | FS | ICL | Metric | Notes |
|---|---|---|---|---|---|
| `truthfulqa/truthful_qa` (multiple_choice) | validation | — | 0 | `truthfulqa_mc2` | single score; built-in 5-QA primer |

MC2 = normalized probability mass on the set of true answers. Higher is better.
The two-key variant `truthfulqa:mc` (reports both `truthfulqa_mc1` + `truthfulqa_mc2`) is also available.

```bash
truthfulqa:mc2:cf|0
```

### CyberMetric + SecQA

| Task | Dataset | Eval | FS | ICL |
|---|---|---|---|---|
| `cybermetric` | `tihanyin/CyberMetric` | train | train | 0 |
| `secqa:v1` / `secqa:v2` | `zefang-liu/secqa` | test | test | 0 |

```bash
cybermetric:cf|0
cybermetric:mcf|0
cybermetric:mcf_em|0
secqa:v1:cf|0
secqa:v2:cf|0
secqa:cf|0                   # both versions
```

### MaScQA

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `heegyu/mascqa` | test | test | 0 | choices embedded in question text |

```bash
mascqa:cf|0
mascqa:mcf|0
mascqa:mcf_em|0
```

### FormationEval

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `AlmazErmilov/FormationEval` | test | test | 0 | petroleum engineering |

```bash
formationeval:cf|0
formationeval:mcf|0
formationeval:mcf_em|0
```

### TeleQnA

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `netop/TeleQnA` | test | test | 0 | **gated** — must be pre-cached |

```bash
teleqna:cf|0
teleqna:mcf|0
teleqna:mcf_em|0
```

### LAB-Bench (TableQA)

| Dataset | Eval | FS | ICL | Notes |
|---|---|---|---|---|
| `futurehouse/LAB-Bench` (TableQA) | train | train | 0 | biology; tables provided as images (text-only variant) |

```bash
labbench:cf|0
labbench:mcf|0
labbench:mcf_em|0
```

### TitanEval English Domain Tasks

Loaded from local parquet (`data/titaneval/`). Source: TitanEval-MCQ benchmark suite.
Each task has its own file: `tasks/{task}.py` (e.g., `tasks/preflight.py`).
All expose `:cf` (acc + acc_norm_char + **BPB merged**), `:mcf` (acc + acc_norm_char), `:mcf_em` (exact match, greedy decode).

> **Few-shot note:** These tasks have **test split only** — no dedicated few-shot split exists.
> The config sets `few_shots_split="test"`, `few_shots_select="random_sampling"` to allow
> CLI-level overrides, but running with `|N` (N > 0) draws examples from the test set itself
> (leakage risk). **Recommended: use `|0` (0-shot) for all TitanEval tasks.**

| Task | Domain | Rows | ICL | Notes |
|---|---|---|---|---|
| `preflight` | Aviation safety | 300 | 0 | |
| `chembench` | Chemistry (analytical/organic/physical) | 2,542 | 0 | |
| `esgenius` | ESG / sustainability | 1,136 | 0 | |
| `xfinbench` | Finance (cross-lingual, EN subset) | 588 | 0 | 4 rows filtered (missing choices) |
| `geobench` | Geoscience | 1,390 | 0 | |

```bash
preflight:cf|0
preflight:mcf|0
preflight:mcf_em|0
chembench:cf|0
chembench:mcf|0
esgenius:cf|0
xfinbench:cf|0
geobench:cf|0
```

---

## 6. English GenQA Tasks

All tasks in this section expose two variants:
- `:gen|N` → `f1`, `em` (normalized: lowercase + remove punctuation on both gold and prediction)
- `:bpb|N` → `target_bpb` (decoupled: same query, scores only the first gold continuation)

Exception: `drop:gen` uses `Metrics.drop` (span/number/date-aware normalization), not the standard normalized EM/F1.

**OLMo Gen2MC note:** OLMo's Base Main Suite reformulates DROP, CoQA, SQuAD, NaturalQs, and Jeopardy as MC (Gen2MC, §A.4.2). Our codebase implements **gen-only** variants for all of these except Jeopardy — which has a dedicated MC dataset as `jeopardy_mc` (§5).

| Task | Dataset | Eval | FS | ICL | gen_size | Notes |
|---|---|---|---|---|---|---|
| `coqa` | `EleutherAI/coqa` | validation | eval | 0 | 50 | OLMo Gen2MC |
| `drop` | `lighteval/drop_harness` | validation | train | 5 | 100 | OLMo Gen2MC; uses `Metrics.drop` |
| `jeopardy` | `soldni/jeopardy` | train | train | 5 | 50 | OLMo Gen2MC; MC form → `jeopardy_mc` (§5) |
| `natural_questions` | `google-research-datasets/nq_open` | validation | train | 5 | 50 | OLMo Gen2MC |
| `squad` | `allenai/squad` (v1.1) | validation | train | 5 | 50 | OLMo Gen2MC |
| `squad_v2` | `rajpurkar/squad_v2` (answerable-only) | validation | train | 5 | 200 | OLMo Gen2MC |
| `triviaqa` | `mandarjoshi/trivia_qa` (rc.nocontext) | validation | train | 5 | 20 | |
| `popqa` | `akariasai/PopQA` | test | test | 5 | 8 | |
| `wikifact:{subset}` (81) | `lighteval/wikifact` | test | test | 5 | 8 | |
| `simpleqa` | `lighteval/SimpleQA` | test | few_shot | 0 | 50 | |

```bash
# CoQA (0-shot conversation QA)
coqa:gen
coqa:bpb

# DROP (discrete reasoning)
drop:gen|5
drop:bpb|5

# Jeopardy (gen form; MC form → jeopardy_mc in §5)
jeopardy:gen|5
jeopardy:bpb|5

# NaturalQuestions
natural_questions:gen|5
natural_questions:bpb|5

# SQuAD v1.1
squad:gen|5
squad:bpb|5

# SQuAD v2 (unanswerable questions excluded via hf_filter)
squad_v2:gen|5
squad_v2:bpb|5

# TriviaQA
triviaqa:gen|5
triviaqa:bpb|5

# PopQA
popqa:gen|5
popqa:bpb|5

# WikiFact (81 relation subsets)
wikifact:gen|5             # all 81 subsets
wikifact:bpb|5             # all 81 subsets
wikifact:author:gen|5      # single subset

# SimpleQA
simpleqa:gen|0
simpleqa:bpb|0
```

**Prompt formats:**
- CoQA: `Passage: {story}\n\nFinal question:\n\nQuestion: {q}\nAnswer:` — stop `["\n\n"]`
- DROP: `Passage: {passage}\nQuestion: {question}\nAnswer:` — stop `["\n\n", "Passage:", "Question:"]`
- Jeopardy: `Category: {cat}\nQuestion: {q}\nAnswer:` — stop `["\n\n", "Question:", "Category:"]`
- NaturalQs: `Question: {question}\nAnswer:` — stop `["Question:", "Q:", "\n\n"]`
- SQuAD: `Title: {title}\n\nBackground: {context}\n\nQuestion: {question}\n\nAnswer:` — stop `["Title:", "\n\n"]`
- SQuAD v2: QA template (same prompt as SQuAD) — stop `["\n", "Question:", "question:"]`
- TriviaQA: `Question: {question}\nAnswer:` — stop `["\n", ".", ","]`; all aliases as gold
- PopQA: `{question} ` — stop `["\n"]`; `possible_answers` list as gold
- WikiFact: `{question} ` — stop `["\n"]`; `references` list as gold
- SimpleQA: `Question: {question}\nAnswer:` — stop `["\n"]`

---

## 7. Lambada & Basic Skills

These tasks use rank-choice or cloze formulations. BPB is **decoupled** (separate `:bpb` config with a different prompt) — it is **not** merged into `:cf`.

### Lambada

| Dataset | Eval | ICL | Config | Metrics |
|---|---|---|---|---|
| `cimec/lambada` | test | 0 | `lambada:cf` | `acc_norm` (char-norm) |
| | | | `lambada:bpb` | `target_bpb` (decoupled) |
| | | | `lambada:standard_cloze` | `target_perplexity` |
| `EleutherAI/lambada_openai` | test | 0 | `lambada:openai_cloze` | `target_perplexity` |

`lambada:cf` uses a distractor format (gold last word vs 3 sampled distractors, scored by char-norm logprob).
`lambada:bpb` scores the full passage continuation directly.

```bash
lambada:cf                    # rank-choice, acc_norm
lambada:bpb                   # BPB decoupled
```

### Basic Skills

| Dataset | Eval | ICL | Config | Metrics |
|---|---|---|---|---|
| `allenai/basic-skills` | validation | 5 | `basic_skills:{subset}:cf` | `acc` (unnorm), `acc_norm` (token-norm) |
| | | | `basic_skills:{subset}:mcf` | `acc` (unnorm), `acc_norm` (char-norm) |
| | | | `basic_skills:{subset}:bpb` | `target_bpb` (decoupled) |

`:cf` and `:mcf` use multi-choice formats with distractors.
`:bpb` uses a single-choice (gold-only) prompt.

```bash
basic_skills:cf|5             # all 6 subsets, acc + acc_norm (token)
basic_skills:mcf|5            # all 6 subsets, acc + acc_norm (char)
basic_skills:bpb|5            # all 6 subsets, BPB decoupled
basic_skills:arithmetic:cf|5  # single subset
basic_skills:arithmetic:bpb|5
```

**Subsets (6):** `arithmetic`, `string_operations`, `coding`, `logical_reasoning`,
`common_knowledge`, `pattern`.

---

## 8. Skipped / Unavailable Tasks

**Skipped (data issues):**
- `nuclearqa` — all choices empty in titaneval parquets; source dataset not found
- `ctibench` — 0 valid rows after filtering (all choices are empty strings in titaneval parquet)

---

## 9. Metric Definitions

**`:cf`** — log p(answer_i | prompt) for each candidate; argmax. Reports:
- `acc` — argmax correct
- `acc_norm` — argmax correct with per-char length normalization
- `target_bpb` — `-log₂ p(gold) / bytes_utf8(gold)`, lower is better
  (merged into `:cf` for English MC QA tasks §5 only; decoupled for lambada, basic_skills)

**`:mcf`** — log p(" A" | prompt), log p(" B" | prompt), … argmax. Reports `acc`, `acc_norm`.

**`:mcf_em`** — greedy decode 1 token, exact-match against gold label string. Reports `em`.

**`:bpb`** (standalone) — single gold continuation, no ranking. Reports `target_bpb`.
Used for code, math, and as decoupled companion to `:gen` or `:cf`.

**`:gen`** — greedy decode up to `generation_size` tokens. Scored with:
- `qa_em` — exact match after `harness_triviaqa_normalizer` (lowercase + remove punctuation) on both gold and prediction; aggregates `max` over all gold aliases
- `qa_f1` — bag-of-words F1 after same normalization; aggregates `max` over all gold aliases
- Exception: `drop:gen` uses `Metrics.drop` (handles number spans, dates, multi-span answers)
