#!/usr/bin/env python
"""Count English-only Encyclo-K examples by discipline."""

from collections import Counter

from datasets import load_dataset

from lighteval.tasks.tasks.encyclo_k import _ENCYCLO_K_DISCIPLINES, _has_english_discipline


DATASET_REPO = "m-a-p/Encyclo-K"
DATASET_REVISION = "0f81da9652b9c4d7bad73ce9576d81901d9441f2"


def main():
    dataset = load_dataset(
        DATASET_REPO,
        "default",
        revision=DATASET_REVISION,
        split="test",
    )

    disciplines = set(_ENCYCLO_K_DISCIPLINES)
    counts = Counter()
    answer_count_distribution = Counter()
    random_guess_score_sums = Counter()

    for row in dataset:
        discipline = row["discipline"]
        if discipline in disciplines and _has_english_discipline(row, discipline):
            number_of_answers = len(row["options"])
            counts[discipline] += 1
            answer_count_distribution[number_of_answers] += 1
            random_guess_score_sums[discipline] += 1 / number_of_answers

    print(f"{'Discipline':<15} {'English examples':>16} {'Random guess':>14}")
    print(f"{'-' * 15} {'-' * 16} {'-' * 14}")
    for discipline in _ENCYCLO_K_DISCIPLINES:
        random_guess_score = random_guess_score_sums[discipline] / counts[discipline]
        print(f"{discipline:<15} {counts[discipline]:>16,} {random_guess_score:>13.2%}")

    total_count = sum(counts.values())
    overall_random_guess_score = sum(random_guess_score_sums.values()) / total_count
    print(f"{'-' * 15} {'-' * 16} {'-' * 14}")
    print(f"{'Total':<15} {total_count:>16,} {overall_random_guess_score:>13.2%}")

    print("\nNumber of answer choices among English examples")
    print(f"{'Choices':>7} {'Examples':>16}")
    print(f"{'-' * 7} {'-' * 16}")
    for number_of_answers in sorted(answer_count_distribution):
        print(f"{number_of_answers:>7} {answer_count_distribution[number_of_answers]:>16,}")
    print(f"{'-' * 7} {'-' * 16}")
    print(f"{'Total':>7} {sum(answer_count_distribution.values()):>16,}")


if __name__ == "__main__":
    main()
