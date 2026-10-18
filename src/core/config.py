"""
Pipeline configuration.

The YAML file is parsed into frozen dataclasses and fully validated up front, so a
malformed or unsupported configuration fails with a descriptive ``ConfigurationError``
before any Spark work starts. Unknown fields are rejected rather than silently ignored.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Optional, Sequence, Union

import yaml

from src.core.exceptions import ConfigurationError
from src.quality.validator import VALIDATION_CHECKS
from src.transforms.functions import AGGREGATION_FUNCTIONS, EXPRESSION_FUNCTIONS
from src.transforms.operators import CONDITION_OPERATORS, LOGICAL_OPERATORS, Predicate

DEFAULT_DATE_FORMATS = ("yyyy-MM-dd", "MM/dd/yyyy")
_DECIMAL_TYPE = re.compile(r"^decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)$", re.IGNORECASE)


# --------------------------------------------------------------------------- models
@dataclass(frozen=True)
class ColumnSpec:
    """A source column and its target type: ``string``, ``date`` or ``decimal(p,s)``."""

    name: str
    type_name: str = "string"
    precision: Optional[int] = None
    scale: Optional[int] = None


@dataclass(frozen=True)
class SourceConfig:
    """CSV source. ``columns`` is the explicit schema; if empty, the header is used (strings)."""

    path: str
    columns: tuple[ColumnSpec, ...] = ()


@dataclass(frozen=True)
class NormalizationConfig:
    """``trim_columns=None`` trims every string column; ``date_formats`` are tried in order."""

    trim_columns: Optional[tuple[str, ...]] = None
    uppercase_columns: tuple[str, ...] = ()
    date_formats: tuple[str, ...] = DEFAULT_DATE_FORMATS


@dataclass(frozen=True)
class RuleSpec:
    """A validation rule: ``check`` must hold for ``column`` (with optional ``value``)."""

    column: str
    check: str
    value: Any = None


@dataclass(frozen=True)
class DataQualityConfig:
    """Input data quality. Exact duplicates are always removed after the rules."""

    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    input_rules: tuple[RuleSpec, ...] = ()


@dataclass(frozen=True)
class CurrencyConversionConfig:
    target_currency: str
    rates_path: str
    price_column: str = "price"
    currency_column: str = "currency"
    date_column: str = "trade_date"


@dataclass(frozen=True)
class ConditionSpec:
    """A filter condition: a comparison on ``column`` or a logical group of ``conditions``."""

    operator: str
    column: Optional[str] = None
    value: Any = None
    conditions: tuple["ConditionSpec", ...] = ()


@dataclass(frozen=True)
class Expression:
    """
    One node of an expression tree. ``kind`` is ``column`` (``value`` = column name),
    ``literal`` (``value`` = the scalar) or ``function`` (``value`` = function name, with
    child ``inputs`` and ``params``).
    """

    kind: str
    value: Any = None
    inputs: tuple["Expression", ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterStep:
    name: str
    condition: ConditionSpec
    type: ClassVar[str] = "filter"


@dataclass(frozen=True)
class ExpressionStep:
    name: str
    output_column: str
    expression: Expression
    type: ClassVar[str] = "expression"


@dataclass(frozen=True)
class AggregationSpec:
    column: str
    function: str
    alias: str


@dataclass(frozen=True)
class AggregateStep:
    name: str
    group_by: tuple[str, ...]
    aggregations: tuple[AggregationSpec, ...]
    type: ClassVar[str] = "aggregate"


TransformStep = Union[FilterStep, ExpressionStep, AggregateStep]


@dataclass(frozen=True)
class OutputConfig:
    """Final output: the columns to keep (in order) and the single CSV file to write."""

    columns: tuple[str, ...]
    path: str = "output/processed_instruments.csv"


@dataclass(frozen=True)
class PipelineConfig:
    """A validated pipeline configuration."""

    name: str
    version: str
    source: SourceConfig
    transformations: tuple[TransformStep, ...]
    output: OutputConfig
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    currency_conversion: Optional[CurrencyConversionConfig] = None
    validation_rules: tuple[RuleSpec, ...] = ()

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load and validate configuration from a YAML file."""
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {path}") from None
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Any) -> "PipelineConfig":
        """Validate an already-loaded configuration mapping."""
        raw = _mapping(
            raw,
            "configuration",
            required=("pipeline", "source", "transformations", "output"),
            optional=("data_quality", "currency_conversion", "validation"),
        )
        info = _mapping(raw["pipeline"], "pipeline", required=("name", "version"))
        version = info["version"]
        if isinstance(version, bool) or not isinstance(version, (str, int, float)):
            raise ConfigurationError(f"pipeline.version: expected a string or number, got {version!r}")

        steps = tuple(
            parse_transform_step(item, f"transformations[{i}]")
            for i, item in enumerate(_list(raw["transformations"], "transformations", non_empty=True))
        )
        names = [step.name for step in steps]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ConfigurationError(f"transformations: duplicate step name(s) {duplicates}")

        validation = _mapping(raw.get("validation", {}), "validation", optional=("rules",))
        config = cls(
            name=_string(info["name"], "pipeline.name"),
            version=str(version),
            source=_parse_source(raw["source"]),
            transformations=steps,
            output=_parse_output(raw["output"]),
            data_quality=_parse_data_quality(raw.get("data_quality", {})),
            currency_conversion=(
                _parse_currency_conversion(raw["currency_conversion"])
                if "currency_conversion" in raw
                else None
            ),
            validation_rules=_parse_rules(validation.get("rules", []), "validation.rules"),
        )
        config._check_source_references()
        return config

    def _check_source_references(self) -> None:
        """Cross-section checks against the declared source schema (when one is declared)."""
        if not self.source.columns:
            return
        types = {column.name: column.type_name for column in self.source.columns}
        normalization = self.data_quality.normalization
        for key, names in (
            ("trim_columns", normalization.trim_columns or ()),
            ("uppercase_columns", normalization.uppercase_columns),
        ):
            for name in names:
                if types.get(name) != "string":
                    raise ConfigurationError(
                        f"data_quality.normalization.{key}: '{name}' must be a string column "
                        f"declared in source.columns"
                    )
        conversion = self.currency_conversion
        if conversion is not None:
            for key in ("price_column", "currency_column", "date_column"):
                if getattr(conversion, key) not in types:
                    raise ConfigurationError(
                        f"currency_conversion.{key}: '{getattr(conversion, key)}' "
                        f"is not declared in source.columns"
                    )


# ------------------------------------------------------------------- parse helpers
def _mapping(
    value: Any, where: str, *, required: Sequence[str] = (), optional: Optional[Sequence[str]] = ()
) -> Mapping[str, Any]:
    """
    Require a mapping. Unless ``optional`` is None (free-form mapping), only the
    ``required`` and ``optional`` keys are allowed; required ones must be present.
    """
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{where}: expected a mapping, got {type(value).__name__}")
    missing = [key for key in required if key not in value]
    if missing:
        raise ConfigurationError(f"{where}: missing required field(s) {missing}")
    if optional is not None:
        unknown = sorted(set(value) - set(required) - set(optional))
        if unknown:
            raise ConfigurationError(
                f"{where}: unknown field(s) {unknown}; allowed: {sorted({*required, *optional})}"
            )
    return value


def _list(value: Any, where: str, *, non_empty: bool = False) -> list:
    if not isinstance(value, list):
        raise ConfigurationError(f"{where}: expected a list, got {type(value).__name__}")
    if non_empty and not value:
        raise ConfigurationError(f"{where}: must not be empty")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{where}: expected a non-empty string, got {value!r}")
    return value.strip()


def _string_list(value: Any, where: str, *, non_empty: bool = False) -> tuple[str, ...]:
    items = tuple(_string(item, f"{where}[{i}]") for i, item in enumerate(_list(value, where, non_empty=non_empty)))
    if len(set(items)) != len(items):
        raise ConfigurationError(f"{where}: contains duplicate entries")
    return items


def _choice(value: Any, allowed: Sequence[str], where: str) -> str:
    text = _string(value, where)
    if text not in allowed:
        raise ConfigurationError(f"{where}: unsupported value {text!r}; supported: {list(allowed)}")
    return text


def _only_value(raw: Mapping[str, Any], key: str, allowed: str, where: str) -> None:
    """Validate an option that currently supports a single value (and is not stored)."""
    if key in raw:
        _choice(raw[key], (allowed,), f"{where}.{key}")


def _checked_value(predicate: Predicate, name: str, value: Any, where: str) -> Any:
    try:
        return predicate.validate(name, value)
    except ValueError as exc:
        raise ConfigurationError(f"{where}: {exc}") from exc


# ------------------------------------------------------------------ section parsers
def _parse_column(name: str, raw_type: Any, where: str) -> ColumnSpec:
    if name.startswith("_"):
        raise ConfigurationError(f"{where}: column names starting with '_' are reserved")
    text = _string(raw_type, where).lower()
    if text in ("string", "date"):
        return ColumnSpec(name, text)
    match = _DECIMAL_TYPE.match(text)
    if match:
        precision, scale = int(match.group(1)), int(match.group(2))
        if not (1 <= precision <= 38 and 0 <= scale <= precision):
            raise ConfigurationError(
                f"{where}: invalid decimal precision/scale in {raw_type!r} "
                "(need 1 <= precision <= 38 and 0 <= scale <= precision)"
            )
        return ColumnSpec(name, "decimal", precision, scale)
    raise ConfigurationError(
        f"{where}: unsupported column type {raw_type!r}; use string, date or decimal(p,s)"
    )


def _parse_source(raw: Any) -> SourceConfig:
    raw = _mapping(raw, "source", required=("type", "path"), optional=("columns",))
    _choice(raw["type"], ("csv",), "source.type")
    columns: tuple[ColumnSpec, ...] = ()
    if "columns" in raw:
        declared = _mapping(raw["columns"], "source.columns", optional=None)
        if not declared:
            raise ConfigurationError("source.columns: must not be empty")
        columns = tuple(
            _parse_column(_string(name, "source.columns"), type_name, f"source.columns.{name}")
            for name, type_name in declared.items()
        )
    return SourceConfig(path=_string(raw["path"], "source.path"), columns=columns)


def _parse_rules(raw: Any, where: str) -> tuple[RuleSpec, ...]:
    rules = []
    for i, item in enumerate(_list(raw, where)):
        item_where = f"{where}[{i}]"
        item = _mapping(item, item_where, required=("column", "check"), optional=("value",))
        check = _string(item["check"], f"{item_where}.check")
        if check not in VALIDATION_CHECKS:
            raise ConfigurationError(
                f"{item_where}.check: unsupported check {check!r}; supported: {sorted(VALIDATION_CHECKS)}"
            )
        value = _checked_value(VALIDATION_CHECKS[check].predicate, check, item.get("value"), item_where)
        rules.append(RuleSpec(_string(item["column"], f"{item_where}.column"), check, value))
    return tuple(rules)


def _parse_data_quality(raw: Any) -> DataQualityConfig:
    where = "data_quality"
    raw = _mapping(raw, where, optional=("normalization", "input_rules", "duplicates", "on_failure"))
    if "duplicates" in raw:
        duplicates = _mapping(raw["duplicates"], f"{where}.duplicates", optional=("action",))
        _only_value(duplicates, "action", "remove_and_report", f"{where}.duplicates")
    _only_value(raw, "on_failure", "quarantine", where)

    norm_where = f"{where}.normalization"
    norm = _mapping(
        raw.get("normalization", {}),
        norm_where,
        optional=("trim_columns", "uppercase_columns", "date_formats"),
    )
    trim = norm.get("trim_columns")
    normalization = NormalizationConfig(
        trim_columns=None if trim is None else _string_list(trim, f"{norm_where}.trim_columns"),
        uppercase_columns=_string_list(norm.get("uppercase_columns", []), f"{norm_where}.uppercase_columns"),
        date_formats=(
            _string_list(norm["date_formats"], f"{norm_where}.date_formats", non_empty=True)
            if "date_formats" in norm
            else DEFAULT_DATE_FORMATS
        ),
    )
    return DataQualityConfig(
        normalization=normalization,
        input_rules=_parse_rules(raw.get("input_rules", []), f"{where}.input_rules"),
    )


def _parse_currency_conversion(raw: Any) -> CurrencyConversionConfig:
    where = "currency_conversion"
    raw = _mapping(
        raw,
        where,
        required=("target_currency", "rates_path"),
        optional=("on_missing_rate", "price_column", "currency_column", "date_column"),
    )
    _only_value(raw, "on_missing_rate", "quarantine", where)
    return CurrencyConversionConfig(
        target_currency=_string(raw["target_currency"], f"{where}.target_currency").upper(),
        rates_path=_string(raw["rates_path"], f"{where}.rates_path"),
        price_column=_string(raw.get("price_column", "price"), f"{where}.price_column"),
        currency_column=_string(raw.get("currency_column", "currency"), f"{where}.currency_column"),
        date_column=_string(raw.get("date_column", "trade_date"), f"{where}.date_column"),
    )


def _parse_condition(raw: Any, where: str) -> ConditionSpec:
    raw = _mapping(raw, where, optional=None)
    if "operator" not in raw:
        raise ConfigurationError(f"{where}: missing required field 'operator'")
    operator = _string(raw["operator"], f"{where}.operator")

    if operator in LOGICAL_OPERATORS:
        _mapping(raw, where, required=("operator",), optional=("conditions", "condition"))
        if "condition" in raw:
            if operator != "not" or "conditions" in raw:
                raise ConfigurationError(
                    f"{where}: 'condition' is only valid for 'not' (use 'conditions' otherwise)"
                )
            children = [raw["condition"]]
        elif "conditions" in raw:
            children = _list(raw["conditions"], f"{where}.conditions", non_empty=True)
        else:
            raise ConfigurationError(f"{where}: operator '{operator}' requires 'conditions'")
        if operator == "not" and len(children) != 1:
            raise ConfigurationError(f"{where}: operator 'not' takes exactly one condition")
        return ConditionSpec(
            operator=operator,
            conditions=tuple(
                _parse_condition(child, f"{where}.conditions[{i}]") for i, child in enumerate(children)
            ),
        )

    predicate = CONDITION_OPERATORS.get(operator)
    if predicate is None:
        raise ConfigurationError(
            f"{where}.operator: unsupported operator {operator!r}; "
            f"supported: {sorted([*CONDITION_OPERATORS, *LOGICAL_OPERATORS])}"
        )
    _mapping(raw, where, required=("column", "operator"), optional=("value",))
    return ConditionSpec(
        operator=operator,
        column=_string(raw["column"], f"{where}.column"),
        value=_checked_value(predicate, operator, raw.get("value"), where),
    )


def _parse_operand(raw: Any, where: str) -> Expression:
    raw = _mapping(raw, where, optional=None)
    if "column" in raw:
        _mapping(raw, where, required=("column",))
        return Expression("column", _string(raw["column"], f"{where}.column"))
    if "literal" in raw:
        _mapping(raw, where, required=("literal",))
        value = raw["literal"]
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ConfigurationError(f"{where}.literal: expected a scalar value, got {value!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigurationError(f"{where}.literal: {value!r} is not supported")
        return Expression("literal", value)
    if "function" in raw:
        return _parse_function(raw, where)
    raise ConfigurationError(f"{where}: an input must define one of 'column', 'literal' or 'function'")


def _parse_function(raw: Mapping[str, Any], where: str) -> Expression:
    function = _string(raw.get("function"), f"{where}.function")
    spec = EXPRESSION_FUNCTIONS.get(function)
    if spec is None:
        raise ConfigurationError(
            f"{where}.function: unsupported function {function!r}; supported: {sorted(EXPRESSION_FUNCTIONS)}"
        )
    _mapping(raw, where, required=("function",), optional=("inputs", *spec.params))
    inputs = tuple(
        _parse_operand(item, f"{where}.inputs[{i}]")
        for i, item in enumerate(_list(raw.get("inputs", []), f"{where}.inputs"))
    )
    try:
        spec.check_arity(len(inputs))
    except ValueError as exc:
        raise ConfigurationError(f"{where}: function {function!r} {exc}") from exc
    params = {}
    for name, default in spec.params.items():
        value = raw.get(name, default)
        if type(value) is not type(default):
            raise ConfigurationError(f"{where}.{name}: expected {type(default).__name__}, got {value!r}")
        params[name] = value
    return Expression("function", function, inputs, params)


def _parse_filter(raw: Mapping[str, Any], name: str, where: str) -> FilterStep:
    _mapping(raw, where, required=("name", "type", "condition"))
    return FilterStep(name, _parse_condition(raw["condition"], f"{where}.condition"))


def _parse_expression_step(raw: Mapping[str, Any], name: str, where: str) -> ExpressionStep:
    _mapping(raw, where, required=("name", "type", "output_column", "expression"))
    return ExpressionStep(
        name=name,
        output_column=_string(raw["output_column"], f"{where}.output_column"),
        expression=_parse_function(_mapping(raw["expression"], f"{where}.expression", optional=None), f"{where}.expression"),
    )


def _parse_aggregate(raw: Mapping[str, Any], name: str, where: str) -> AggregateStep:
    _mapping(raw, where, required=("name", "type", "group_by", "aggregations"))
    group_by = _string_list(raw["group_by"], f"{where}.group_by", non_empty=True)
    aggregations = []
    for i, item in enumerate(_list(raw["aggregations"], f"{where}.aggregations", non_empty=True)):
        item_where = f"{where}.aggregations[{i}]"
        item = _mapping(item, item_where, required=("column", "function", "alias"))
        function = _string(item["function"], f"{item_where}.function")
        if function not in AGGREGATION_FUNCTIONS:
            raise ConfigurationError(
                f"{item_where}.function: unsupported aggregation {function!r}; "
                f"supported: {sorted(AGGREGATION_FUNCTIONS)}"
            )
        column = _string(item["column"], f"{item_where}.column")
        if column == "*" and function != "count":
            raise ConfigurationError(f"{item_where}.column: '*' is only valid with 'count'")
        aggregations.append(AggregationSpec(column, function, _string(item["alias"], f"{item_where}.alias")))
    aliases = [a.alias for a in aggregations]
    if len(set(aliases)) != len(aliases):
        raise ConfigurationError(f"{where}.aggregations: aliases must be unique, got {aliases}")
    clashes = sorted(set(aliases) & set(group_by))
    if clashes:
        raise ConfigurationError(f"{where}.aggregations: alias(es) {clashes} clash with group_by columns")
    return AggregateStep(name, group_by, tuple(aggregations))


# Step type -> parser. Keep in sync with ``src.transforms.registry.TRANSFORM_REGISTRY``.
STEP_PARSERS: dict[str, Callable[[Mapping[str, Any], str, str], TransformStep]] = {
    "filter": _parse_filter,
    "expression": _parse_expression_step,
    "aggregate": _parse_aggregate,
}


def parse_transform_step(raw: Any, where: str = "transformations[0]") -> TransformStep:
    """Validate one YAML transformation step and return its typed model."""
    raw = _mapping(raw, where, optional=None)
    for key in ("name", "type"):
        if key not in raw:
            raise ConfigurationError(f"{where}: missing required field '{key}'")
    name = _string(raw["name"], f"{where}.name")
    step_type = _string(raw["type"], f"{where}.type")
    parser = STEP_PARSERS.get(step_type)
    if parser is None:
        raise ConfigurationError(
            f"{where}.type: unsupported transformation type {step_type!r}; supported: {sorted(STEP_PARSERS)}"
        )
    return parser(raw, name, f"{where} ({name})")


def _parse_output(raw: Any) -> OutputConfig:
    raw = _mapping(raw, "output", required=("columns",), optional=("path",))
    path = _string(raw.get("path", OutputConfig.path), "output.path")
    if not path.lower().endswith(".csv"):
        raise ConfigurationError(f"output.path: expected a .csv file path, got {path!r}")
    return OutputConfig(_string_list(raw["columns"], "output.columns", non_empty=True), path)
