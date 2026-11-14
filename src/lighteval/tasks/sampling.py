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

"""Opt-in seeded subsets (titan_eval flag `data.sample_per_group=N`, owner 11-14: "a flag to the number of samples, but never cut by default").

The DEFAULT of every task is its full official set. A task that supports the flag builds its `hf_filter` with `subset_filter(...)`: that returns None
(= the full set, the filter does not exist) unless the run exported `TITAN_EVAL_SAMPLE_PER_GROUP=N` (titan_eval's lighteval adapter does, from the config
key `data.sample_per_group`). With N set, every GROUP of the task (a language, a difficulty level, an OMEGA bucket: each lighteval subtask is one group)
keeps N rows, drawn with `random.Random(seed).sample(sorted unique keys, N)`: the draw depends on the group's seed string and the keys only, so it is the
same on every machine and every run, and the ids are reproducible from the code and the pinned dataset (nothing is stored). A group with at most N rows
keeps all of them.

`apply_dataset_filter` is what `lighteval_task` calls for `hf_filter`: a plain predicate goes to `Dataset.filter`; an object with an `apply(dataset)`
method (the subsets here) takes the whole dataset (a DatasetDict or a Dataset) so that it can see all the rows of a group before it picks.
"""

import os
import random
from typing import Callable, Hashable, Mapping, Sequence


ENV_SAMPLE_PER_GROUP = "TITAN_EVAL_SAMPLE_PER_GROUP"


def sample_per_group() -> int | None:
    """N of the run's `data.sample_per_group` (env TITAN_EVAL_SAMPLE_PER_GROUP), None = the full set. A value that is not a positive integer is an error."""
    raw = os.environ.get(ENV_SAMPLE_PER_GROUP)
    if raw is None or raw.strip() == "":
        return None
    try:
        n = int(raw)
    except ValueError:
        n = 0
    if n < 1:
        raise ValueError(f"{ENV_SAMPLE_PER_GROUP}={raw!r}: data.sample_per_group must be a positive integer (or unset for the full set)")
    return n


class SeededSubset:
    """`hf_filter` object: N rows of each evaluated split, a seeded draw of unique keys (`key(row)`), all rows of a drawn key are kept."""

    def __init__(self, seed: str, n: int, key: Callable[[Mapping], Hashable], splits: Sequence[str]):
        self.seed, self.n, self.key, self.splits = seed, int(n), key, tuple(splits)

    def pick(self, rows: Sequence[Mapping]) -> frozenset:
        """The drawn keys of one group (the same call gives the same keys)."""
        keys = sorted({self.key(r) for r in rows})
        if len(keys) <= self.n:
            return frozenset(keys)
        return frozenset(random.Random(self.seed).sample(keys, self.n))

    def _one(self, ds):
        keep = self.pick(ds)
        return ds.select([i for i, row in enumerate(ds) if self.key(row) in keep])

    def apply(self, dataset):
        if hasattr(dataset, "keys"):                      # a DatasetDict: only the evaluated split(s) are drawn, the rest is not used
            from datasets import DatasetDict

            return DatasetDict({name: (self._one(d) if name in self.splits else d) for name, d in dataset.items()})
        return self._one(dataset)

    def __repr__(self) -> str:
        return f"SeededSubset(seed={self.seed!r}, n={self.n}, splits={list(self.splits)})"


def subset_filter(seed: str, key: Callable[[Mapping], Hashable], splits: Sequence[str]) -> "SeededSubset | None":
    """The `hf_filter` of a group that supports the flag: None (the full set) unless the run asked for N rows per group."""
    n = sample_per_group()
    return None if n is None else SeededSubset(seed, n, key, splits)


def apply_dataset_filter(dataset, flt):
    """`hf_filter` applied to a loaded dataset: a predicate -> `.filter`, an object with `apply` -> its own way."""
    apply = getattr(flt, "apply", None)
    return apply(dataset) if callable(apply) else dataset.filter(flt)
