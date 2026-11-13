"""Fixed held-out few-shot demonstrations for benchmarks that ship only ONE split (base_extra suite).

Not a task module (no TASKS_TABLE). A benchmark with a single evaluation split cannot take its shots from another split, so a
small FIXED slice of that split is held out: it is shown as the demonstrations of every prompt, in a defined order, and is NOT
scored. The scored item count drops by the size of the slice.

The slice is chosen by a deterministic rule (`select_shots`) and written down as constants (row keys) in the task module; the
unit test recomputes the rule on the pinned dataset and checks the constants. At run time a task only needs:
  * `load_rows(...)`   - the pinned split as a list of plain dicts (the same `load_dataset` call as the task, so the offline
                         cache is hit);
  * `rows_by_key(...)` - the held-out rows, in the order they are shown (`LightevalTaskConfig.few_shots_rows`);
  * its prompt function returns None for a held-out row when it is NOT building a demonstration (`line["__few_shots"]` False).

Selection rule (stable across Python / numpy versions: only sha256 and integer comparison):
  rank the rows by sha256(TAG | identity) hex digest, ascending; then
  * `labels` given and `want` None: walk the ranking and take the first row of every label not seen yet ("distinct labels"),
    then fill up with the next rows of the ranking;
  * `want` given (a list of labels, one per shot): for each wanted label, in order, take the first not yet used row of the
    ranking that has this label; a wanted label of None takes the next unused row;
  * the chosen rows are shown in a second pseudo-random order: sha256(TAG | "order" | identity) ascending.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Callable, Hashable, Sequence


TAG = "base-extra-v1"


def digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def content_key(*parts: object) -> str:
    """16-hex content key of a row (sha256 prefix of its text fields); identifies a held-out row without positions."""
    return digest("row", *parts)[:16]


def select_shots(
    identities: Sequence[object],
    labels: Sequence[Hashable] | None = None,
    k: int = 5,
    want: Sequence[Hashable | None] | None = None,
    tag: str = TAG,
) -> list[int]:
    """Positions (into `identities`) of the k held-out rows, in the order they are shown. See the module docstring."""
    ranking = sorted(range(len(identities)), key=lambda i: digest(tag, identities[i]))
    chosen: list[int] = []
    if want is not None:
        if len(want) != k:
            raise ValueError(f"`want` has {len(want)} labels for k={k} shots")
        for wanted in want:
            for i in ranking:
                if i not in chosen and (wanted is None or labels[i] == wanted):
                    chosen.append(i)
                    break
            else:
                raise ValueError(f"no unused row with label {wanted!r}")
    else:
        seen: set = set()
        if labels is not None:
            for i in ranking:
                if labels[i] not in seen:
                    seen.add(labels[i])
                    chosen.append(i)
                    if len(chosen) == k:
                        break
        for i in ranking:
            if len(chosen) == k:
                break
            if i not in chosen:
                chosen.append(i)
    return sorted(chosen, key=lambda i: digest(tag, "order", identities[i]))


@lru_cache(maxsize=None)
def _load_rows(repo: str, subset: str, split: str, revision: str) -> tuple:
    from lighteval.tasks.lighteval_task import load_dataset_pinned

    # The same call (path / name / revision) as lighteval's download_dataset_worker, so the cached files are reused offline.
    return tuple(dict(row) for row in load_dataset_pinned(repo, subset, revision=revision)[split])


def load_rows(repo: str, subset: str, split: str, revision: str) -> list[dict]:
    """The pinned split as a list of plain dicts (cached per process)."""
    return list(_load_rows(repo, subset, split, revision))


def rows_by_key(
    repo: str, subset: str, split: str, revision: str, key_of: Callable[[dict], object], keys: Sequence[object]
) -> list[dict]:
    """The held-out rows with `key_of(row) == key`, in the order of `keys`. Fails loudly when a key is missing or ambiguous."""
    index: dict[object, list[dict]] = {}
    for row in load_rows(repo, subset, split, revision):
        index.setdefault(key_of(row), []).append(row)
    out = []
    for key in keys:
        hits = index.get(key, [])
        if len(hits) != 1:
            raise ValueError(f"{repo}/{subset}/{split}@{revision[:8]}: {len(hits)} rows with key {key!r}, expected exactly 1")
        out.append(hits[0])
    return out
