"""
Expression and aggregation function registries.

Every function is built from native Spark ``Column`` expressions (no UDFs, no ``eval``).
To add a function: write a builder, register it in ``EXPRESSION_FUNCTIONS`` or
``AGGREGATION_FUNCTIONS``, and add tests. The engine does not need to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from functools import reduce
from typing import Any, Callable, Mapping, Sequence

from pyspark.sql import Column
from pyspark.sql import functions as F


def literal(value: Any) -> Column:
    """
    Build a Spark literal from a YAML scalar.

    Floats become exact decimals (``0.1`` -> ``Decimal("0.1")``) so financial arithmetic
    does not pick up binary floating point error.
    """
    if value is None:
        return F.lit(None)
    if isinstance(value, float):
        return F.lit(Decimal(format(Decimal(repr(value)), "f")))
    return F.lit(value)


@dataclass(frozen=True)
class ExpressionFunction:
    """
    A registered expression function.

    Attributes:
        min_args: Minimum number of ``inputs``.
        max_args: Maximum number of ``inputs`` (``None`` for unbounded).
        build: Builds the Spark column from resolved inputs and YAML parameters.
        params: Optional YAML parameters and their defaults (e.g. ``scale`` for round).
    """

    min_args: int
    max_args: int | None
    build: Callable[[Sequence[Column], Mapping[str, Any]], Column]
    params: Mapping[str, Any] = field(default_factory=dict)

    def check_arity(self, count: int) -> None:
        """Raise ``ValueError`` if ``count`` inputs are not accepted."""
        too_few = count < self.min_args
        too_many = self.max_args is not None and count > self.max_args
        if not (too_few or too_many):
            return
        if self.max_args is None:
            expected = f"at least {self.min_args}"
        elif self.min_args == self.max_args:
            expected = f"exactly {self.min_args}"
        else:
            expected = f"{self.min_args} to {self.max_args}"
        raise ValueError(f"expects {expected} input(s), got {count}")


def _add(args: Sequence[Column], _: Mapping[str, Any]) -> Column:
    return reduce(lambda left, right: left + right, args)


def _multiply(args: Sequence[Column], _: Mapping[str, Any]) -> Column:
    return reduce(lambda left, right: left * right, args)


def _subtract(args: Sequence[Column], _: Mapping[str, Any]) -> Column:
    return args[0] - args[1]


def _divide(args: Sequence[Column], _: Mapping[str, Any]) -> Column:
    """Safe division: a null or zero denominator yields null instead of failing."""
    numerator, denominator = args
    return F.when(denominator.isNull() | (denominator == 0), F.lit(None)).otherwise(
        numerator / denominator
    )


EXPRESSION_FUNCTIONS: dict[str, ExpressionFunction] = {
    # Arithmetic. Any null input yields null.
    "add": ExpressionFunction(2, None, _add),
    "subtract": ExpressionFunction(2, 2, _subtract),
    "multiply": ExpressionFunction(2, None, _multiply),
    "divide": ExpressionFunction(2, 2, _divide),
    "abs": ExpressionFunction(1, 1, lambda args, _: F.abs(args[0])),
    "round": ExpressionFunction(
        1, 1, lambda args, params: F.round(args[0], params["scale"]), {"scale": 0}
    ),
    # General / string.
    "coalesce": ExpressionFunction(1, None, lambda args, _: F.coalesce(*args)),
    "upper": ExpressionFunction(1, 1, lambda args, _: F.upper(args[0])),
    "lower": ExpressionFunction(1, 1, lambda args, _: F.lower(args[0])),
    "trim": ExpressionFunction(1, 1, lambda args, _: F.trim(args[0])),
    "concat": ExpressionFunction(1, None, lambda args, _: F.concat(*args)),
    # Time.
    "current_timestamp": ExpressionFunction(0, 0, lambda _a, _p: F.current_timestamp()),
    "current_date": ExpressionFunction(0, 0, lambda _a, _p: F.current_date()),
}


# Aggregations take the input column and return the aggregated column.
# NOTE: ``first`` and ``last`` are not deterministic without an ordering guarantee.
# Use them only for columns known to be constant within each group (e.g. a currency
# already normalised to a single value); never to pick between conflicting values.
AGGREGATION_FUNCTIONS: dict[str, Callable[[Column], Column]] = {
    "sum": F.sum,
    "avg": F.avg,
    "min": F.min,
    "max": F.max,
    "count": F.count,
    "count_distinct": F.countDistinct,
    "first": F.first,
    "last": F.last,
}
