"""
Aggregate transformation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from src.transforms.base import Transform, require_columns
from src.transforms.functions import AGGREGATION_FUNCTIONS

if TYPE_CHECKING:
    from src.core.config import AggregateStep, AggregationSpec


def _build_aggregation(spec: "AggregationSpec") -> Column:
    source = F.lit(1) if spec.column == "*" else F.col(spec.column)
    return AGGREGATION_FUNCTIONS[spec.function](source).alias(spec.alias)


class AggregateTransform(Transform):
    """
    Groups rows and computes the configured aggregations.

    The result contains only the ``group_by`` columns followed by one column per
    aggregation alias.
    """

    def __init__(self, step: "AggregateStep"):
        self.step = step

    def apply(self, df: DataFrame) -> DataFrame:
        needed = list(self.step.group_by) + [
            a.column for a in self.step.aggregations if a.column != "*"
        ]
        require_columns(df, needed, f"aggregate '{self.step.name}'")
        aggregations = [_build_aggregation(spec) for spec in self.step.aggregations]
        return df.groupBy(*self.step.group_by).agg(*aggregations)
