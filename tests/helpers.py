"""
Shared test helpers.
"""
import copy
from typing import Any

from src.core.config import PipelineConfig, parse_transform_step
from src.transforms.registry import build_transform


def make_transform(step: dict[str, Any]):
    """Validate a raw YAML step and build its transformation."""
    return build_transform(parse_transform_step(step))


def minimal_config() -> dict[str, Any]:
    """A small valid configuration (as a fresh dict) that tests can modify."""
    return copy.deepcopy(
        {
            "pipeline": {"name": "test_pipeline", "version": "1"},
            "source": {"type": "csv", "path": "data/input/instruments.csv"},
            "transformations": [
                {
                    "name": "keep_positive",
                    "type": "filter",
                    "condition": {"column": "price", "operator": "gt", "value": 0},
                }
            ],
            "output": {"columns": ["price"]},
        }
    )


def config_from(raw: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig.from_dict(raw)
