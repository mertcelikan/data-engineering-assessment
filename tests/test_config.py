"""
YAML loading and configuration validation tests.
"""
import pytest

from src.core.config import AggregateStep, ExpressionStep, FilterStep, PipelineConfig
from src.core.exceptions import ConfigurationError
from tests.helpers import config_from, minimal_config


def with_step(step: dict) -> dict:
    raw = minimal_config()
    raw["transformations"] = [step]
    return raw


class TestValidConfig:
    def test_loads_assessment_yaml(self):
        config = PipelineConfig.from_yaml("config/pipeline.yaml")

        assert (config.name, config.version) == ("instrument_daily_metrics", "1.0")
        assert [type(step) for step in config.transformations] == [
            FilterStep,
            ExpressionStep,
            AggregateStep,
            ExpressionStep,
            ExpressionStep,
        ]
        assert [c.name for c in config.source.columns] == [
            "instrument_id", "trade_date", "price", "volume", "currency",
        ]  # fmt: skip
        assert config.data_quality.normalization.date_formats == ("yyyy-MM-dd", "MM/dd/yyyy")
        assert config.currency_conversion.target_currency == "USD"
        assert [r.check for r in config.validation_rules] == ["not_null", "positive"]
        assert config.output.path == "output/processed_instruments.csv"

    def test_optional_sections_have_defaults(self):
        config = config_from(minimal_config())

        assert config.currency_conversion is None
        assert config.data_quality.input_rules == ()
        assert config.validation_rules == ()
        assert config.output.path == "output/processed_instruments.csv"

    def test_nested_conditions_and_expressions_are_parsed(self):
        condition = {
            "operator": "and",
            "conditions": [
                {"column": "a", "operator": "between", "value": [1, 5]},
                {"operator": "not", "condition": {"column": "b", "operator": "is_null"}},
            ],
        }
        expression = {
            "function": "round",
            "scale": 2,
            "inputs": [{"function": "multiply", "inputs": [{"column": "p"}, {"literal": 1.1}]}],
        }

        steps = config_from(
            {
                **minimal_config(),
                "transformations": [
                    {"name": "f", "type": "filter", "condition": condition},
                    {"name": "e", "type": "expression", "output_column": "o", "expression": expression},
                ],
            }
        ).transformations

        assert steps[0].condition.conditions[0].value == (1, 5)
        assert steps[0].condition.conditions[1].conditions[0].operator == "is_null"
        assert steps[1].expression.params == {"scale": 2}
        assert steps[1].expression.inputs[0].inputs[1].value == 1.1


class TestFileErrors:
    @pytest.mark.parametrize(
        "content, message",
        [
            (None, "not found"),
            ("pipeline: [unclosed", "Invalid YAML"),
            ("- a\n- b\n", "expected a mapping"),
        ],
    )
    def test_unreadable_files(self, tmp_path, content, message):
        path = tmp_path / "pipeline.yaml"
        if content is not None:
            path.write_text(content)

        with pytest.raises(ConfigurationError, match=message):
            PipelineConfig.from_yaml(str(path))


def _set(path: str, value):
    """A mutation that sets a nested key such as ``source.columns``."""

    def apply(raw):
        *parents, key = path.split(".")
        target = raw
        for parent in parents:
            target = target[parent]
        target[key] = value

    return apply


def _with_trim_on_numeric(raw):
    raw["source"]["columns"] = {"price": "decimal(18,6)"}
    raw["data_quality"] = {"normalization": {"trim_columns": ["price"]}}


INVALID_DOCUMENTS = [
    (lambda raw: raw.pop("source"), "source"),
    (_set("transformation", []), "unknown field"),
    (_set("transformations", []), "must not be empty"),
    (lambda raw: raw.update(transformations=raw["transformations"] * 2), "duplicate step name"),
    (_set("source.type", "parquet"), "source.type"),
    (_set("source.columns", {"price": "int"}), "source.columns.price"),
    (_set("source.columns", {"_hidden": "string"}), "reserved"),
    (_with_trim_on_numeric, "must be a string column"),
    (_set("output.path", "output/result.parquet"), "output.path"),
    (
        _set("currency_conversion", {"target_currency": "USD", "rates_path": "x", "on_missing_rate": "ignore"}),
        "on_missing_rate",
    ),
    (_set("validation", {"rules": [{"column": "a", "check": "bogus"}]}), "unsupported check 'bogus'"),
]


class TestInvalidDocuments:
    @pytest.mark.parametrize("mutate, message", INVALID_DOCUMENTS)
    def test_invalid_document(self, mutate, message):
        raw = minimal_config()
        mutate(raw)

        with pytest.raises(ConfigurationError, match=message):
            config_from(raw)


def _filter(condition: dict) -> dict:
    return {"name": "f", "type": "filter", "condition": condition}


def _expression(expression: dict) -> dict:
    return {"name": "e", "type": "expression", "output_column": "out", "expression": expression}


def _aggregate(aggregations: list) -> dict:
    return {"name": "a", "type": "aggregate", "group_by": ["g"], "aggregations": aggregations}


def _agg(function="sum", alias="x", column="v") -> dict:
    return {"column": column, "function": function, "alias": alias}


A = {"column": "a"}
INVALID_STEPS = [
    ({"name": "x", "type": "join"}, "unsupported transformation type"),
    ({"type": "filter"}, "missing required field 'name'"),
    (_filter({"column": "a", "operator": "approx", "value": 1}), "unsupported operator 'approx'"),
    (_filter({"column": "a", "operator": "gt"}), "requires a 'value'"),
    (_filter({"operator": "and"}), "requires 'conditions'"),
    (
        _filter(
            {
                "operator": "or",
                "conditions": [
                    {"column": "a", "operator": "gt", "value": 1},
                    {"operator": "not", "condition": {"column": "b", "operator": "nope", "value": 1}},
                ],
            }
        ),
        r"conditions\[1\]\.conditions\[0\]",  # errors name the nested location
    ),
    (_expression({"function": "sqrt", "inputs": [A]}), "unsupported function 'sqrt'"),
    (_expression({"function": "divide", "inputs": [A]}), "exactly 2"),
    (_expression({"function": "round", "inputs": [A], "scale": "2"}), "expected int"),
    (_aggregate([_agg("median")]), "unsupported aggregation 'median'"),
    (_aggregate([_agg(alias="g")]), "clash"),
]


class TestInvalidSteps:
    @pytest.mark.parametrize("step, message", INVALID_STEPS)
    def test_invalid_step(self, step, message):
        with pytest.raises(ConfigurationError, match=message):
            config_from(with_step(step))
