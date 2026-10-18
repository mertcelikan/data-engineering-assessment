"""
Core pipeline components.

Public names are resolved lazily so that lightweight modules (``src.core.exceptions``)
can be imported from anywhere without pulling in the whole engine and creating import
cycles.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .engine import PipelineEngine

__all__ = ["PipelineEngine", "PipelineConfig"]


def __getattr__(name: str):
    if name == "PipelineEngine":
        from .engine import PipelineEngine

        return PipelineEngine
    if name == "PipelineConfig":
        from .config import PipelineConfig

        return PipelineConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
