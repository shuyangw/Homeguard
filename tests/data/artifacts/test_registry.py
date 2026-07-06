from datetime import date
from pathlib import Path
import pytest
from src.data.artifacts.base import ArtifactBuilder
from src.data.artifacts import registry


class _A(ArtifactBuilder):
    name = "a"
    output_subdir = "a"
    def inputs(self): return []
    def build(self, start, end): return self.output_path()


class _B(ArtifactBuilder):
    name = "b"
    output_subdir = "b"
    def inputs(self): return ["a"]
    def build(self, start, end): return self.output_path()


def test_resolve_order_is_topological():
    reg = registry.Registry()
    reg.register(_A())
    reg.register(_B())
    assert reg.resolve_order(["b"]) == ["a", "b"]


def test_output_path_under_artifacts_fx(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.artifacts.base.get_local_storage_dir", lambda: tmp_path)
    p = _A().output_path()
    assert p == tmp_path / "artifacts" / "fx" / "a"


def test_missing_dependency_raises():
    reg = registry.Registry()
    reg.register(_B())
    with pytest.raises(KeyError):
        reg.resolve_order(["b"])


def test_all_builders_returns_registered_builders():
    reg = registry.Registry()
    builder = _A()
    reg.register(builder)
    builders = reg.all_builders()
    assert builders == {"a": builder}


def test_all_builders_returns_a_copy():
    reg = registry.Registry()
    reg.register(_A())
    builders = reg.all_builders()
    builders["b"] = _B()
    assert "b" not in reg.all_builders()
