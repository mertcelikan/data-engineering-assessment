"""
Transform base.
"""
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from pyspark.sql import DataFrame

from src.core.exceptions import SchemaError


def require_columns(df: DataFrame, columns: Iterable[str], context: str) -> None:
    """Raise ``SchemaError`` if any of ``columns`` is missing from ``df`` (case-insensitive)."""
    available = {name.lower() for name in df.columns}
    missing = sorted({c for c in columns if c.lower() not in available})
    if missing:
        raise SchemaError(f"{context}: column(s) {missing} not found; available: {df.columns}")


class Transform(ABC):
    """Base class for transformations."""

    @abstractmethod
    def apply(self, df: DataFrame) -> DataFrame:
        """Apply the transformation."""
        pass

    def rejected_records(self, df: DataFrame) -> Optional[DataFrame]:
        """
        Rows of ``df`` that ``apply`` would drop, or ``None`` if the transform never
        drops rows. Lets the engine count removed records without knowing the
        transform type.
        """
        return None
