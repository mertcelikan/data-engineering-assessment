"""
Expression transformation.
"""
from __future__ import annotations

from typing import Iterator

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from src.core.config import Expression, ExpressionStep
from src.transforms.base import Transform, require_columns
from src.transforms.functions import EXPRESSION_FUNCTIONS, literal


def build_expression(node: Expression) -> Column:
    """Resolve a column reference, literal or (nested) function call to a Spark column."""
    if node.kind == "column":
        return F.col(node.value)
    if node.kind == "literal":
        return literal(node.value)
    function = EXPRESSION_FUNCTIONS[node.value]
    return function.build([build_expression(child) for child in node.inputs], node.params)


def _referenced_columns(node: Expression) -> Iterator[str]:
    if node.kind == "column":
        yield node.value
    for child in node.inputs:
        yield from _referenced_columns(child)


class ExpressionTransform(Transform):
    """Adds (or replaces) ``output_column`` with the result of an expression."""

    def __init__(self, step: ExpressionStep):
        self.step = step

    def apply(self, df: DataFrame) -> DataFrame:
        require_columns(
            df, _referenced_columns(self.step.expression), f"expression '{self.step.name}'"
        )
        return df.withColumn(self.step.output_column, build_expression(self.step.expression))
