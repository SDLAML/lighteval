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

"""lighteval._fast_metadata: the stat-light packages_distributions() must equal the standard library's.

Runs under pytest or directly (`python tests/unit/utils/test_fast_metadata.py`); standard library only. The
comparison on the live environment stats every installed file the standard-library way, so it takes minutes on a cold
network file system (that is what the replacement avoids).
"""

import importlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from lighteval import _fast_metadata


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_site(root: Path) -> None:
    """A fake site-packages: declared top_level.txt, RECORD-only (inferred), RECORD with a missing file, egg-info."""
    # declared: top_level.txt wins, RECORD is not consulted
    d = root / "fmdeclared-1.0.dist-info"
    _write(d / "METADATA", "Metadata-Version: 2.1\nName: fmdeclared\nVersion: 1.0\n")
    _write(d / "top_level.txt", "fm_decl_a\nfm_decl_b\n")
    _write(d / "RECORD", "fm_unlisted/__init__.py,,\n")
    # inferred from RECORD: a package, a single module, a non-importable name, an outside path, a package with no file
    d = root / "fminferred-2.0.dist-info"
    _write(d / "METADATA", "Metadata-Version: 2.1\nName: fm-inferred\nVersion: 2.0\n")
    _write(root / "fm_pkg" / "__init__.py")
    _write(root / "fm_pkg" / "sub" / "x.py")
    _write(root / "fm_mod.py")
    _write(root / "fm_gone" / "__init__.py")  # exists on disk only so the RECORD below can list it, removed after
    _write(
        d / "RECORD",
        "\n".join(
            [
                "fm_pkg/__init__.py,sha256=x,0",
                "fm_pkg/sub/x.py,sha256=x,0",
                "fm_pkg/never_written.py,sha256=x,0",
                "fm_mod.py,sha256=x,0",
                "fm_gone/__init__.py,sha256=x,0",
                "fm_gone/other.py,sha256=x,0",
                "fm_data.pth,sha256=x,0",
                "../../bin/fm_tool,sha256=x,0",
                "fminferred-2.0.dist-info/METADATA,,",
                "fminferred-2.0.dist-info/RECORD,,",
                "",
            ]
        ),
    )
    (root / "fm_gone" / "__init__.py").unlink()  # every file of fm_gone is now missing
    (root / "fm_gone").rmdir()
    # egg-info without RECORD: the standard-library path
    e = root / "fmegg-3.0.egg-info"
    _write(e / "PKG-INFO", "Metadata-Version: 1.1\nName: fmegg\nVersion: 3.0\n")
    _write(e / "SOURCES.txt", "fm_egg_pkg/__init__.py\n")
    _write(root / "fm_egg_pkg" / "__init__.py")


def _with_site(fn):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_site(root)
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        try:
            return fn()
        finally:
            sys.path.remove(str(root))
            importlib.invalidate_caches()


def test_synthetic_site_matches_stdlib_and_expectations():
    def check():
        fast = _fast_metadata.packages_distributions()
        std = _fast_metadata.STDLIB_PACKAGES_DISTRIBUTIONS()
        assert fast == std, {k: (fast.get(k), std.get(k)) for k in set(fast) | set(std) if fast.get(k) != std.get(k)}
        assert fast["fm_decl_a"] == ["fmdeclared"] and fast["fm_decl_b"] == ["fmdeclared"]
        assert "fm_unlisted" not in fast  # top_level.txt wins over RECORD
        assert fast["fm_pkg"] == ["fm-inferred"] and fast["fm_mod"] == ["fm-inferred"]
        assert "fm_gone" not in fast  # all its files are missing
        assert not any("." in k or k == ".." for k in fast)  # fm_data.pth, ../../bin/fm_tool
        assert fast["fm_egg_pkg"] == ["fmegg"]  # egg-info: the standard path

    _with_site(check)


def test_live_environment_matches_stdlib():
    t0 = time.time()
    fast = _fast_metadata.packages_distributions()
    t_fast = time.time() - t0
    std = _fast_metadata.STDLIB_PACKAGES_DISTRIBUTIONS()
    assert fast == std, sorted(k for k in set(fast) | set(std) if fast.get(k) != std.get(k))
    assert len(fast) > 10
    print(f"live environment: {len(fast)} names equal to the standard library's, fast version {t_fast:.2f} s")


def _patched_function_module(env_extra: dict) -> str:
    env = dict(os.environ)
    env.pop("LIGHTEVAL_STDLIB_PACKAGES_DISTRIBUTIONS", None)
    env.update(env_extra)
    code = "import importlib.metadata as m; import lighteval; print(m.packages_distributions.__module__)"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_import_installs_and_opt_out_keeps_stdlib():
    assert _patched_function_module({}) == "lighteval._fast_metadata"
    assert _patched_function_module({"LIGHTEVAL_STDLIB_PACKAGES_DISTRIBUTIONS": "1"}) == "importlib.metadata"
    assert _patched_function_module({"LIGHTEVAL_STDLIB_PACKAGES_DISTRIBUTIONS": "0"}) == "lighteval._fast_metadata"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
