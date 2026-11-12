#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Usage: bash scripts/run_beyondbench.sh [BASE_MODEL_PATH] [MODEL_NAME]
BASE_MODEL_PATH="${1:-${BASE_MODEL_PATH:-/e/data1/trustllmd/filatov1/hf-models}}"
MODEL_NAME="${2:-${MODEL_NAME:-LFM2-8B-A1B}}"

SUITE="easy" # NB: only easy is supported for now
DATAPOINTS=100
SEED=42

# Default list sizes for each group.
GENERAL_SIZES="${GENERAL_SIZES:-8,16}"
# Exercise both odd- and even-length median calculations.
MEDIAN_SIZES="${MEDIAN_SIZES:-9,16}"
ELEMENT_FREQUENCY_SIZES="${ELEMENT_FREQUENCY_SIZES:-8,16}"
# Keep products small enough for BeyondBench's float-based answer scorer.
MULTIPLICATION_SIZES="${MULTIPLICATION_SIZES:-2,4}"
PAIRED_LIST_SIZES="${PAIRED_LIST_SIZES:-5,10}"
SET_OPERATIONS_SIZES="${SET_OPERATIONS_SIZES:-6,12}"

OUTPUT_ROOT="./results/beyond_bench/${MODEL_NAME}"

# Shared evaluation settings
eval_args=(
  --suite "${SUITE}"
  --datapoints "${DATAPOINTS}"
  --seed "${SEED}"
  --backend vllm
  --trust-remote-code
  --store-details
)

# Run a group of tasks sharing the same list size.
# Each group has its own output directory to avoid overwriting results.
run_group() {
  local group="$1"
  local list_sizes="$2"
  shift 2

  local task_args=()
  local task

  for task in "$@"; do
    task_args+=(--tasks "$task")
  done

  local dir_suffix="${list_sizes//,/_}"
  local range_args=()
  if [[ "$group" == "set_operations" ]]; then
    # Increase overlap between the two short lists.
    range_args=(--range-min -20 --range-max 20)
    # Keep these results separate from previous runs with the wider range.
    dir_suffix+="_range_-20_20"
  elif [[ "$group" == "element_frequency" ]]; then
    # Make repeated values common enough to exercise frequency counting.
    range_args=(--range-min -10 --range-max 10)
    dir_suffix+="_range_-10_10"
  fi
  local output_dir="${OUTPUT_ROOT}/${group}_${dir_suffix}"

  printf '\n=== %s: %s tasks, list sizes %s ===\n' \
    "$group" "$#" "$list_sizes"

  ### Base model eval
  python -u "${SCRIPT_DIR}/beyondbench_eval.py" evaluate \
    "${eval_args[@]}" \
    "${range_args[@]}" \
    "${task_args[@]}" \
    --list-sizes "${list_sizes}" \
    --model-id "${BASE_MODEL_PATH}/${MODEL_NAME}" \
    --no-chat-template \
    --temperature 0 \
    --top-p 1 \
    --max-tokens 4096 \
    --output-dir "${output_dir}"

  # ### Chat model eval
  # python -u "${SCRIPT_DIR}/beyondbench_eval.py" evaluate \
  #   "${eval_args[@]}" \
  #   "${range_args[@]}" \
  #   "${task_args[@]}" \
  #   --list-sizes "${list_sizes}" \
  #   --model-id "${BASE_MODEL_PATH}/${MODEL_NAME}" \
  #   --temperature 0.6 \
  #   --top-p 0.95 \
  #   --max-tokens 32768 \
  #   --output-dir "${output_dir}"
}

# --------------------------------------------------
# GROUP 1: General list tasks
# Default lengths: 8, 16
# --------------------------------------------------

general_tasks=(
  sorting
  sum
  odd_count
  even_count
  find_maximum
  find_minimum
  mean
  mode
  second_maximum
  range
  index_of_maximum
  count_negative
  count_unique
  max_adjacent_difference
  count_greater_than_previous
  sum_of_max_indices
  count_palindromic
  longest_increasing_subsequence
  sum_of_digits
  count_perfect_squares
  alternating_sum
  count_multiples
  local_maxima_count
  weighted_sum
  running_average
  parity_check
  cumulative_sum
  reverse_list
  rotate_list
  moving_average
  second_minimum
  variance
  standard_deviation
)

run_group "general" "${GENERAL_SIZES}" "${general_tasks[@]}"

# --------------------------------------------------
# GROUP 2: Median
# Default lengths: 9, 16 (odd and even)
# --------------------------------------------------

run_group "median" "${MEDIAN_SIZES}" \
  median

# --------------------------------------------------
# GROUP 3: Element frequency
# Default lengths: 8, 16
# Number range: [-10, 10], applied by run_group to encourage repeated values.
# --------------------------------------------------

run_group "element_frequency" "${ELEMENT_FREQUENCY_SIZES}" \
  element_frequency

# --------------------------------------------------
# GROUP 4: Multiplication
# Default lengths: 2, 4 factors
# --------------------------------------------------

run_group "multiplication" "${MULTIPLICATION_SIZES}" \
  multiplication

# --------------------------------------------------
# GROUP 5: Two-list tasks
# Default lengths: 5, 10 elements per list
# --------------------------------------------------

run_group "paired_lists" "${PAIRED_LIST_SIZES}" \
  interleave_lists \
  dot_product

# --------------------------------------------------
# GROUP 6: Set operations
# Default lengths: 6, 12 elements per list
# Number range: [-20, 20], applied by run_group to increase overlap.
# --------------------------------------------------

run_group "set_operations" "${SET_OPERATIONS_SIZES}" \
  set_intersection \
  set_difference

# --------------------------------------------------
# GROUP 7: Fixed-input tasks
# Absolute difference always uses 2 numbers.
# Other tasks ignore --list-sizes.
# --------------------------------------------------

run_group "fixed_inputs" "2" \
  absolute_difference \
  comparison \
  division \
  subtraction

echo "Completed easy-suite evaluations."
echo "Results: ${OUTPUT_ROOT}"