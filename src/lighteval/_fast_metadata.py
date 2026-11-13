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

"""A cheap `importlib.metadata.packages_distributions()` (standard library only).

Why: `transformers/utils/import_utils.py` calls `importlib.metadata.packages_distributions()` when it is imported, and
the standard library (3.13) answers by `stat()`ing EVERY file listed in the RECORD of EVERY installed distribution
(about 120 000 files in the eval environment). On the cluster's network file system that is pure I/O wait: measured
on the login node 45-130 s of wall time for 2.6 s of CPU on a cold metadata cache (about 3 s when it is warm). Every
process that imports transformers pays it again: the lighteval CLI and each spawned data-parallel vLLM worker (four
of them at once, all waiting for the same metadata).

`packages_distributions()` here returns the same mapping, but a top-level name of a distribution is confirmed by
`stat()`ing the files of that name until one exists (usually the first), instead of all files: a few hundred stats
instead of ~120 000 (0.3 s). Distributions without a RECORD (egg-info) go through the standard library path.

`install()` is called from `lighteval/__init__.py`, i.e. before anything can import transformers. Set
LIGHTEVAL_STDLIB_PACKAGES_DISTRIBUTIONS=1 to keep the standard library function.
"""

import csv
import importlib.metadata as _im
import inspect
import os
from collections import defaultdict


STDLIB_PACKAGES_DISTRIBUTIONS = _im.packages_distributions  # kept for the tests and for the opt-out


def _top_level_name(path) -> str:
    """Same rule as importlib.metadata._get_toplevel_name: the first path component, or the module name of a
    single-file entry."""
    top, *rest = path.parts
    return top if rest else (inspect.getmodulename(path) or str(path))


def _inferred_top_levels(dist) -> set:
    """Importable top-level names of `dist` from its files (what importlib.metadata infers when top_level.txt is
    missing), checking that at least one file of each name exists."""
    record = dist.read_text("RECORD")
    if not record:  # egg-info and the like: the standard (slow, rare) path
        return {n for n in map(_top_level_name, dist.files or ()) if "." not in n}
    by_name: dict = {}
    for row in csv.reader(record.splitlines()):
        if not row:
            continue
        path = _im.PackagePath(row[0])
        name = _top_level_name(path)
        if "." in name:  # not importable (foo.dist-info, foo.pth, ..)
            continue
        by_name.setdefault(name, []).append(path)
    names = set()
    for name, paths in by_name.items():
        for path in paths:
            if dist.locate_file(path).exists():
                names.add(name)
                break
    return names


def packages_distributions() -> dict:
    """Mapping top-level import name -> distribution names (same result as importlib.metadata's)."""
    pkg_to_dist = defaultdict(list)
    for dist in _im.distributions():
        declared = (dist.read_text("top_level.txt") or "").split()
        for pkg in declared or _inferred_top_levels(dist):
            pkg_to_dist[pkg].append(dist.metadata["Name"])
    return dict(pkg_to_dist)


def install() -> None:
    """Replace importlib.metadata.packages_distributions (idempotent; opt out with the environment variable)."""
    if os.environ.get("LIGHTEVAL_STDLIB_PACKAGES_DISTRIBUTIONS", "") not in ("", "0"):
        return
    _im.packages_distributions = packages_distributions
