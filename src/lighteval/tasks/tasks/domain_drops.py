"""Row filter for the base_science domain tasks whose data have broken or unanswerable rows (owner decisions 2026-11-13).

Not a task module (no TASKS_TABLE). chembench / xfinbench / geobench read a local parquet (`titaneval_local`) whose rows cannot be
repaired in place, so the task DROPS them with a written rule. Each task module has a pure function `drop_reason(row)` that returns
a short reason code (or None = keep); `KeepRows` turns it into the `hf_filter` of the task. The reason codes are what the count in
the task docstring, the registry notes and the review page come from (agents/reward_checks/domain_fixes_checks.py prints them).
The same function can later run in a data-preparation script that writes the kept rows (or a `drop_reason` column) to a prepared file;
nothing in the task would change except where the rows come from.

The filter also removes the dropped rows from the few-shot pool (the pool is the same filtered split), so a broken item is never a
demonstration either.

Cache key: `datasets` names the filtered Arrow file after a hash of the filter object. A plain function is hashed by its NAME, so a
changed rule would silently keep the OLD cached file (agents/DONT.md, mascqa 11-12; it happened again while this module was written:
a regex edited under an unchanged name served the old row set). `KeepRows` therefore stores a digest of everything the rule is made of
(`parts`: the regex patterns, the key set, the dropped field value), so the cache key follows the rule by itself; still bump the task
`version` and the `rule` text when the rule changes.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Callable, Iterable, Mapping


class KeepRows:
    """`hf_filter` object: a row is kept when `reason(row)` is None. `rule` is a version string of the drop rule; `parts` are the
    things the rule is made of (compiled patterns, strings, a key set): their digest is part of the datasets cache key."""

    def __init__(self, rule: str, reason: Callable[[Mapping], str | None], parts: Iterable = ()):
        self.rule = rule
        self.reason = reason
        text = "\x1f".join(str(getattr(p, "pattern", p)) + str(getattr(p, "flags", "")) for p in parts)
        self.digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def __call__(self, row: Mapping) -> bool:
        return self.reason(row) is None

    def counts(self, rows: Iterable[Mapping]) -> Counter:
        """Reason code -> number of rows ("kept" for the kept ones)."""
        return Counter(self.reason(row) or "kept" for row in rows)

    def __repr__(self) -> str:
        return f"KeepRows({self.rule!r}, digest={self.digest})"
