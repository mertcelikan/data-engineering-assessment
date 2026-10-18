"""
Condition operator registry shared by filters and validation rules.

Operators build native Spark ``Column`` expressions; no SQL strings are evaluated.
Null semantics follow Spark: a comparison involving null is null, which filters treat
as "not matching" and validation rules treat as a failure.
"""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from pyspark.sql import Column

from src.transforms.functions import literal

LOGICAL_OPERATORS = ("and", "or", "not")

_SCALAR_TYPES = (str, int, float, bool)


class ValueKind(Enum):
    """Shape of the ``value`` an operator expects."""

    NONE = "no value"
    SCALAR = "a single scalar value"
    LIST = "a non-empty list of scalar values"
    RANGE = "a list of exactly two scalar values [low, high]"


@dataclass(frozen=True)
class Predicate:
    """A column predicate: expected ``value`` shape plus the Spark expression builder."""

    kind: ValueKind
    build: Callable[[Column, Any], Column]
    text_only: bool = False

    def validate(self, name: str, value: Any) -> Any:
        """
        Check ``value`` against this predicate's expected shape.

        Returns the normalised value (lists become tuples) or raises ``ValueError``.
        """
        if self.kind is ValueKind.NONE:
            if value is not None:
                raise ValueError(f"operator '{name}' does not accept a value")
            return None
        if value is None:
            raise ValueError(
                f"operator '{name}' requires a 'value': {self.kind.value}"
                + (" (use is_null / is_not_null to test for null)" if name in ("eq", "neq") else "")
            )
        if self.kind is ValueKind.SCALAR:
            self._check_scalar(name, value)
            return value
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"operator '{name}' expects {self.kind.value}, got {value!r}")
        for item in value:
            self._check_scalar(name, item)
        if self.kind is ValueKind.LIST:
            if not value:
                raise ValueError(f"operator '{name}' expects {self.kind.value}, got an empty list")
        elif len(value) != 2:
            raise ValueError(f"operator '{name}' expects {self.kind.value}, got {len(value)} value(s)")
        elif _is_number(value[0]) and _is_number(value[1]) and value[0] > value[1]:
            raise ValueError(f"operator '{name}' expects low <= high, got {list(value)}")
        return tuple(value)

    def _check_scalar(self, name: str, value: Any) -> None:
        if value is None or not isinstance(value, _SCALAR_TYPES):
            raise ValueError(f"operator '{name}' expects scalar values, got {value!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"operator '{name}' does not accept {value!r}")
        if self.text_only and not isinstance(value, str):
            raise ValueError(f"operator '{name}' expects a string value, got {value!r}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _comparison(op: Callable[[Column, Column], Column]) -> Predicate:
    return Predicate(ValueKind.SCALAR, lambda column, value: op(column, literal(value)))


CONDITION_OPERATORS: dict[str, Predicate] = {
    "gt": _comparison(operator.gt),
    "gte": _comparison(operator.ge),
    "lt": _comparison(operator.lt),
    "lte": _comparison(operator.le),
    "eq": _comparison(operator.eq),
    "neq": _comparison(operator.ne),
    "in": Predicate(
        ValueKind.LIST, lambda column, values: column.isin(*[literal(v) for v in values])
    ),
    "not_in": Predicate(
        ValueKind.LIST, lambda column, values: ~column.isin(*[literal(v) for v in values])
    ),
    "between": Predicate(
        ValueKind.RANGE, lambda column, bounds: column.between(literal(bounds[0]), literal(bounds[1]))
    ),
    "is_null": Predicate(ValueKind.NONE, lambda column, _: column.isNull()),
    "is_not_null": Predicate(ValueKind.NONE, lambda column, _: column.isNotNull()),
    "contains": Predicate(ValueKind.SCALAR, lambda column, v: column.contains(v), text_only=True),
    "starts_with": Predicate(ValueKind.SCALAR, lambda column, v: column.startswith(v), text_only=True),
    "ends_with": Predicate(ValueKind.SCALAR, lambda column, v: column.endswith(v), text_only=True),
}
