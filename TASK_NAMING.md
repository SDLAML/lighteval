# Task Naming Conventions

Every task name ends with an explicit metric-type suffix. There are no bare/default names.

## Suffix Reference


| Suffix    | Metric type                 | `generation_size`      | Metrics reported                | Description                                                                  |
| --------- | --------------------------- | ---------------------- | ------------------------------- | ---------------------------------------------------------------------------- |
| `:mcf_em` | Greedy generation (MC only) | `1`–`5`                | `exact_match`                   | MC tasks only: generate label token, compare with EM                         |
| `:cf`     | Completion formulation      | `-1` (logprob only)    | `acc`, `acc_norm`, `target_bpb` | Score full answer text via log p(choice|context); BPB merged in for MC tasks |
| `:mcf`    | Multiple-choice formulation | `-1` (logprob only)    | `acc`, `acc_norm`               | Score label tokens only ( `A`, `B`, …)                                       |
| `:bpb`    | Bits-per-byte (standalone)  | `-1` (logprob only)    | `target_bpb`                    | Used only where CF is not applicable (MATH, free-form GenQA)                 |


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

No fixed answer choices — CF/MCF are not applicable. BPB uses the gold answer as a standalone continuation.


| Task                | `:bpb` metric |
| ------------------- | ------------- |
| `coqa`              | `target_bpb`  |
| `drop`              | `target_bpb`  |
| `jeopardy`          | `target_bpb`  |
| `natural_questions` | `target_bpb`  |
| `squad_v2`          | `target_bpb`  |


### Math

BPB uses a compact 4-shot OLMo-style prompt.


| Task                        | `:bpb` metric |
| --------------------------- | ------------- |
| `math:{subset}` (7 subsets) | `target_bpb`  |


Subsets: `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus`.

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

**`:mcf_em`** — greedy decode (temperature=0), compare output to gold with exact_match, F1, or task-specific metric.

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