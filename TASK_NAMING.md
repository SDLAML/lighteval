# Task Naming Conventions

Every task name ends with an explicit metric-type suffix. There are no bare/default names.

## Evaluation Suites

### Table 1 — BPB Prepare Suite

Tasks used to measure BPB during pre-training. All tasks run in logprob mode (`generation_size=-1`). CF tasks also produce `acc` and `acc_norm` in the same pass.

```
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
mmlu:cf|5             # expands to all 57 subsets
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
squad_v2:bpb|5
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
mmlu:mcf|5          # expands to all 57 subsets (STEM + Humanities + Social Sci + Other)
med_mcqa:mcf|5
med_qa:mcf|5
sciq:mcf|5
commonsenseqa:mcf|5
piqa:mcf|5
siqa:mcf|5
```

#### GenQA / RC (completion/generation)

```
hellaswag:cf|5          # RC per-char norm
winogrande:cf|5         # RC unnormalized
lambada:cf              # RC per-char norm, 0-shot
basic_skills:cf|5      # RC per-token norm, expands to all 6 subsets
drop:gen|5
jeopardy:gen|5
natural_questions:gen|5
squad_v2:gen|5
coqa:gen                # 0-shot
```

#### Held-out Suite

```
mmlu_pro:mcf|5      # or just: mmlu_pro|5
bigbench_hard|3     # expands to all 27 BBH subsets
```

---

## Suffix Reference


| Suffix    | Metric type                 | `generation_size`   | Metrics reported                | Description                                                                  |
| --------- | --------------------------- | ------------------- | ------------------------------- | ---------------------------------------------------------------------------- |
| `:mcf_em` | Greedy generation (MC only) | `1`–`5`             | `exact_match`                   | MC tasks only: generate label token, compare with EM                         |
| `:cf`     | Completion formulation      | `-1` (logprob only) | `acc`, `acc_norm`, `target_bpb` | Score full answer text via log p(choice|context); BPB merged in for MC tasks |
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
```

## Task inventory

### MMLU

Task names follow `mmlu:{subset}:{metric}` — consistent with all other tasks.


| Superset                | Expands to                                | # tasks      |
| ----------------------- | ----------------------------------------- | ------------ |
| `mmlu`                  | all subsets × all metrics                 | 171 (57 × 3) |
| `mmlu:cf`               | `mmlu:{subset}:cf` for all 57 subsets     | 57           |
| `mmlu:mcf_em`           | `mmlu:{subset}:mcf_em` for all 57 subsets | 57           |
| `mmlu:mcf`              | `mmlu:{subset}:mcf` for all 57 subsets    | 57           |
| `mmlu:abstract_algebra` | all 3 variants for that subset            | 3            |
| `mmlu_redux`            | all redux subsets × all metrics           | 171          |
| `mmlu_redux:cf`         | all redux subsets with CF                 | 57           |


Each `:cf` task reports `{acc, acc_norm, target_bpb}`. There are no standalone `:bpb` tasks for MMLU.

### Multiple-choice QA tasks

For all MC tasks, BPB is embedded in `:cf`. The full evaluation suite per task is `:cf` + `:mcf` + `:mcf_em`.


| Task                         | `:cf` metrics      | `:mcf` metrics | `:mcf_em` metrics | gen_size |
| ---------------------------- | ------------------ | -------------- | ----------------- | -------- |
| `arc:challenge` / `arc:easy` | acc, acc_norm, bpb | acc, acc_norm  | em                | 1        |
| `hellaswag`                  | acc, acc_norm, bpb | acc, acc_norm  | em                | 1        |
| `winogrande`                 | acc, acc_norm, bpb | acc, acc_norm  | em                | 1        |
| `commonsenseqa`              | acc, acc_norm, bpb | acc, acc_norm  | em                | 1        |
| `piqa`                       | acc, acc_norm, bpb | acc, acc_norm  | em                | 1        |
| `siqa`                       | acc, acc_norm, bpb | acc, acc_norm  | em                | 1        |
| `sciq`                       | acc, acc_norm, bpb | acc, acc_norm  | em                | 1        |
| `med_mcqa`                   | acc, acc_norm, bpb | acc, acc_norm  | em                | 5        |
| `med_qa`                     | —                  | acc, acc_norm  | em                | —        |


`med_qa` has no `:cf` variant (open-ended options, no fixed candidate text). 5-shot from train split.

### Single-answer completion tasks

These tasks have a single correct answer text (no distractors), so MCF is not applicable.


| Task                                | `:cf` metrics         |
| ----------------------------------- | --------------------- |
| `lambada`                           | acc_norm (char), bpb  |
| `basic_skills:{subset}` (6 subsets) | acc_norm (token), bpb |


**Lambada special cloze variants** (logprob perplexity only):


| Task                     | Dataset                     | Prompt               | Metrics |
| ------------------------ | --------------------------- | -------------------- | ------- |
| `lambada:standard_cloze` | `cimec/lambada`             | `{context} ____. ->` | ppl     |
| `lambada:openai_cloze`   | `EleutherAI/lambada_openai` | `{context} ____. ->` | ppl     |


### Free-form GenQA tasks

These tasks have free-form gold answers. Two evaluation modes are available:

- `**:bpb`** — logprob-only; measures `p(gold | prompt)`, no generation. Lower is better.
- `**:gen**` — actual greedy decoding; scored with F1 + EM against the gold answer.


| Task                    | `:bpb` (BPB) | `:gen` metrics | `:gen` gen_size | ICL shots |
| ----------------------- | ------------ | -------------- | --------------- | --------- |
| `coqa:bpb`              | `target_bpb` | —              | —               | 0         |
| `coqa:gen`              | —            | `f1`, `em`     | 50              | 0         |
| `drop:bpb`              | `target_bpb` | —              | —               | 5         |
| `drop:gen`              | —            | `f1`, `em`     | 100             | 5         |
| `jeopardy:bpb`          | `target_bpb` | —              | —               | 5         |
| `jeopardy:gen`          | —            | `f1`, `em`     | 50              | 5         |
| `natural_questions:bpb` | `target_bpb` | —              | —               | 5         |
| `natural_questions:gen` | —            | `f1`, `em`     | 50              | 5         |
| `squad_v2:bpb`          | `target_bpb` | —              | —               | 5         |
| `squad_v2:gen`          | —            | `f1`, `em`     | 50              | 5         |


`drop:gen` uses `Metrics.drop` (the standard DROP F1 metric with number normalization). The others use `Metrics.f1_score` + `Metrics.exact_match`.

### Math

Two evaluation modes:

- `**:bpb**` — logprob over gold reference solution (4-shot OLMo-style prompt). Lower is better.
- `**:gen**` — CoT generation, scored with `Metrics.expr_gold_metric` (extracts and compares math expressions).


| Task                        | `:bpb` metric | `:gen` metric      | `:gen` gen_size | ICL shots |
| --------------------------- | ------------- | ------------------ | --------------- | --------- |
| `math:{subset}` (7 subsets) | `target_bpb`  | `extractive_match` | 1024            | 4         |


Subsets: `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus`.

### CoT generation tasks (no logprob variants)

These tasks use greedy generation with chain-of-thought prompting. No `:bpb`/`:cf`/`:mcf` variants.


| Task                          | Dataset                  | ICL | gen_size | Metric                         | Subsets |
| ----------------------------- | ------------------------ | --- | -------- | ------------------------------ | ------- |
| `gsm8k`                       | `openai/gsm8k`           | 8   | 512      | `expr_gold_metric`             | —       |
| `gsm_symbolic:main`           | `apple/GSM-Symbolic`     | 8   | 512      | `expr_gold_metric`             | —       |
| `gsm_symbolic:p1`             | `apple/GSM-Symbolic`     | 8   | 512      | `expr_gold_metric`             | —       |
| `gsm_symbolic:p2`             | `apple/GSM-Symbolic`     | 8   | 512      | `expr_gold_metric`             | —       |
| `math_500`                    | `HuggingFaceH4/MATH-500` | 0   | 1024     | `expr_gold_metric`             | —       |
| `bigbench_hard:{subset}` (27) | `lukaemon/bbh`           | 3   | 1024     | `em` (after answer extraction) | 27      |


**BigBench Hard answer extraction**: the generated chain-of-thought is searched for `"the answer is <X>"` (case-insensitive); `<X>` is extracted and compared against the gold target with exact match. Stop sequences: `["</s>", "Q", "\n\n"]`.

**BBH subsets** (27): `boolean_expressions`, `causal_judgement`, `date_understanding`, `disambiguation_qa`, `dyck_languages`, `formal_fallacies`, `geometric_shapes`, `hyperbaton`, `logical_deduction_five_objects`, `logical_deduction_seven_objects`, `logical_deduction_three_objects`, `movie_recommendation`, `multistep_arithmetic_two`, `navigate`, `object_counting`, `penguins_in_a_table`, `reasoning_about_colored_objects`, `ruin_names`, `salient_translation_error_detection`, `snarks`, `sports_understanding`, `temporal_sequences`, `tracking_shuffled_objects_five_objects`, `tracking_shuffled_objects_seven_objects`, `tracking_shuffled_objects_three_objects`, `web_of_lies`, `word_sorting`.

### Ruler (special case)

Ruler uses 2-part names `ruler_{length}:{subset}` with a custom metric (`Metrics.ruler_match`). No `:cf`/`:bpb` etc. suffixes.


| Query                    | Expands to                                                |
| ------------------------ | --------------------------------------------------------- |
| `ruler_4096`             | all subsets for length 4096                               |
| `ruler_4096:niah`        | all `ruler_4096:niah_*` subtasks (regex prefix expansion) |
| `ruler_4096:niah_single` | all `ruler_4096:niah_single_*` subtasks                   |
| `ruler_4096:vt`          | exact single task                                         |


Requires `TOKENIZER_PATH` env var to be set — `TASKS_TABLE` is empty otherwise.

## Metric definitions

`**:mcf_em`** — greedy decode (temperature=0), compare output to gold with exact_match, F1, or task-specific metric.

`**:cf**` (completion formulation) — prompt ends with `"Answer:"`, score the full candidate answer text:

```
score_i = log p(answer_i | prompt)
```

Prediction = argmax. Normalizations applied: per-char (`LogProbCharNorm`) → `acc_norm`. For MC tasks, BPB is also computed from the gold choice's logprob in the same pass.

`**:mcf**` (multiple-choice formulation) — prompt shows labeled options (A/B/C/D), score only the label token:

```
score_i = log p(" A" | prompt)   # or " B", " C", " D"
```

Prediction = argmax label score. Reports `acc` (unnormalized) and `acc_norm` (char-normalized).

`**:bpb**` (bits-per-byte, standalone) — no choice ranking, gold continuation only:

```
BPB = -log2 p(gold | prompt) / bytes_utf8(gold)
```

Lower is better. Used only for MATH and free-form GenQA where CF is not applicable. For all MC tasks, BPB is reported as part of `:cf` instead.

`**:gen**` (greedy generation) — autoregressively decode up to `generation_size` tokens (temperature=0), then score the generated text:

```
y = argmax_v p(v | prompt, y_<t)   # greedy, per token
```

For GenQA tasks: scored with token-level F1 (partial credit) and exact match against gold answer. For math/CoT tasks: scored with `expr_gold_metric` (extracts and normalises math expressions before comparison).