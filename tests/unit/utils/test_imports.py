# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.

from importlib.metadata import PackageNotFoundError

import pytest
from packaging.requirements import Requirement

from lighteval.utils import imports


@pytest.fixture(autouse=True)
def clear_package_availability_cache():
    imports.is_package_available.cache_clear()
    yield
    imports.is_package_available.cache_clear()


@pytest.mark.parametrize("package", ["vllm", Requirement("vllm>=0.10.0")])
@pytest.mark.parametrize(
    ("installed", "expected"),
    [("1.0.0.dev20261030+cu132", True), ("0.19.1", True), ("0.9.0.dev1", False)],
)
def test_installed_backend_versions(monkeypatch, package, installed, expected):
    monkeypatch.setattr(imports, "required_dependencies", lambda: ({"vllm": Requirement("vllm>=0.10.0")}, {}))
    monkeypatch.setattr(imports, "version", lambda name: installed)

    assert imports.is_package_available(package) is expected


def test_development_backend_still_respects_version_bounds(monkeypatch):
    monkeypatch.setattr(imports, "version", lambda name: "1.0.0.dev20261030+cu132")

    assert not imports.is_package_available(Requirement("vllm>=0.10.0,<0.20.0"))


def test_missing_backend_is_unavailable(monkeypatch):
    def missing_version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(imports, "version", missing_version)

    assert not imports.is_package_available(Requirement("vllm>=0.10.0"))
