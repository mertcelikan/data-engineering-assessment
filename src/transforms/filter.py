"""
Filter transformation.
"""
from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Iterator, Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from src.transforms.base import Transform, require_columns
from src.transforms.operators import CONDITION_OPERATORS

if TYPE_CHECKING:
    from src.core.config import ConditionSpec, FilterStep


def build_condition(spec: "ConditionSpec") -> Column:
    """Build the Spark boolean expression for a (possibly nested) condition."""
    if spec.operator == "and":
        return reduce(operator.and_, (build_condition(child) for child in spec.conditions))
    if spec.operator == "or":
        return reduce(operator.or_, (build_condition(child) for child in spec.conditions))
    if spec.operator == "not":
        return ~build_condition(spec.conditions[0])
    return CONDITION_OPERATORS[spec.operator].build(F.col(spec.column), spec.value)


def _condition_columns(spec: "ConditionSpec") -> Iterator[str]:
    if spec.column is not None:
        yield spec.column
    for child in spec.conditions:
        yield from _condition_columns(child)


class FilterTransform(Transform):
    """Keeps the rows that satisfy a condition; null comparisons do not match."""

    def __init__(self, step: "FilterStep"):
        self.step = step

    def apply(self, df: DataFrame) -> DataFrame:
        self._check_columns(df)
        return df.filter(build_condition(self.step.condition))

    def rejected_records(self, df: DataFrame) -> Optional[DataFrame]:
        self._check_columns(df)
        matches = F.coalesce(build_condition(self.step.condition), F.lit(False))
        return df.filter(~matches)

    def _check_columns(self, df: DataFrame) -> None:
        require_columns(
            df, _condition_columns(self.step.condition), f"filter '{self.step.name}'"
        )
