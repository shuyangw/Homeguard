from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from src.settings import get_local_storage_dir


class ArtifactBuilder(ABC):
    name: str = ""
    output_subdir: str = ""
    REQUIRES_KEY: str | None = None

    @abstractmethod
    def inputs(self) -> list[str]:
        ...

    @abstractmethod
    def build(self, start: date, end: date) -> Path:
        ...

    def output_path(self) -> Path:
        return get_local_storage_dir() / "artifacts" / "fx" / self.output_subdir
