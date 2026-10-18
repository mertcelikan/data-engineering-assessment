"""
Transformation registry.

Maps the YAML ``type`` of a transformation step to the class that executes it, so the
engine never needs to know about concrete transformation types. To add a type: write a
``Transform`` subclass, register it here, and add a step parser in ``src.core.config``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.exceptions import ConfigurationError
from src.transforms.aggregate import AggregateTransform
from src.transforms.base import Transform
from src.transforms.expression import ExpressionTransform
from src.transforms.filter import FilterTransform

if TYPE_CHECKING:
    from src.core.config import TransformStep

TRANSFORM_REGISTRY: dict[str, type[Transform]] = {
    "filter": FilterTransform,
    "expression": ExpressionTransform,
    "aggregate": AggregateTransform,
}


def build_transform(step: "TransformStep") -> Transform:
    """Instantiate the transformation for a validated configuration step."""
    try:
        transform_class = TRANSFORM_REGISTRY[step.type]
    except KeyError:
        raise ConfigurationError(
            f"Transformation '{step.name}': no implementation registered for type "
            f"'{step.type}'; registered: {sorted(TRANSFORM_REGISTRY)}"
        ) from None
    return transform_class(step)
