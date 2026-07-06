from __future__ import annotations
from src.data.artifacts.base import ArtifactBuilder

RAW_FEEDS = {"minute", "quotes", "fred", "oil", "equity_index", "holidays", "calendar"}


class Registry:
    def __init__(self) -> None:
        self._builders: dict[str, ArtifactBuilder] = {}

    def register(self, builder: ArtifactBuilder) -> ArtifactBuilder:
        self._builders[builder.name] = builder
        return builder

    def get_builder(self, name: str) -> ArtifactBuilder:
        return self._builders[name]

    def all_builders(self) -> dict[str, ArtifactBuilder]:
        return dict(self._builders)

    def resolve_order(self, names: list[str]) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(n: str, stack: tuple[str, ...]) -> None:
            if n in seen:
                return
            if n in stack:
                raise ValueError(f"cycle at {n}")
            b = self._builders.get(n)
            if b is None:
                raise KeyError(f"unknown builder: {n}")
            for dep in b.inputs():
                if dep in RAW_FEEDS:
                    continue
                if dep not in self._builders:
                    raise KeyError(f"unknown builder dependency: {dep}")
                visit(dep, stack + (n,))
            seen.add(n)
            order.append(n)

        for name in names:
            visit(name, ())
        return order


_DEFAULT = Registry()
register = _DEFAULT.register
get_builder = _DEFAULT.get_builder
resolve_order = _DEFAULT.resolve_order
all_builders = _DEFAULT.all_builders
