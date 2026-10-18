"""
Unit tests for the Filter, Expression and Aggregate transformations.
"""
from datetime import datetime
from decimal import Decimal

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DecimalType, TimestampType

from src.core.config import STEP_PARSERS
from src.core.exceptions import SchemaError
from src.transforms.base import Transform
from src.transforms.registry import TRANSFORM_REGISTRY
from tests.helpers import make_transform


def values(df, column):
    return [row[column] for row in df.collect()]


def ids(df):
    return sorted(values(df, "id"))


# ----------------------------------------------------------------------------- filter
@pytest.fixture(scope="module")
def trades(spark: SparkSession):
    rows = [
        ("a", 10, "USD", 100),
        ("b", 50, "EUR", 200),
        ("c", 100, "GBP", None),
        ("d", 200, None, 50),
        ("e", -5, "USD", 0),
    ]
    return spark.createDataFrame(rows, "id string, price int, currency string, volume int")


def filter_step(**condition):
    return {"name": "f", "type": "filter", "condition": condition}


class TestFilterTransform:
    @pytest.mark.parametrize(
        "column, operator, value, expected",
        [
            ("price", "gt", 50, ["c", "d"]),
            ("price", "between", [50, 100], ["b", "c"]),  # inclusive
            ("currency", "in", ["USD", "EUR"], ["a", "b", "e"]),
            ("currency", "not_in", ["USD", "EUR"], ["c"]),  # a null never matches
            ("currency", "is_null", None, ["d"]),
        ],
    )
    def test_operators(self, trades, column, operator, value, expected):
        step = filter_step(column=column, operator=operator, **({} if value is None else {"value": value}))

        assert ids(make_transform(step).apply(trades)) == expected

    def test_nested_logic(self, trades):
        step = filter_step(
            operator="and",
            conditions=[
                {
                    "operator": "or",
                    "conditions": [
                        {"column": "currency", "operator": "eq", "value": "USD"},
                        {"column": "currency", "operator": "eq", "value": "EUR"},
                    ],
                },
                {"operator": "not", "condition": {"column": "price", "operator": "lt", "value": 0}},
            ],
        )

        assert ids(make_transform(step).apply(trades)) == ["a", "b"]

    def test_decimal_literals_are_compared_exactly(self, spark):
        df = spark.createDataFrame(
            [("a", Decimal("0.10")), ("b", Decimal("0.11"))], "id string, price decimal(10,2)"
        )

        assert ids(make_transform(filter_step(column="price", operator="gt", value=0.1)).apply(df)) == ["b"]

    def test_rejected_records_are_the_complement_and_include_nulls(self, trades):
        transform = make_transform(filter_step(column="volume", operator="gte", value=100))

        assert ids(transform.apply(trades)) == ["a", "b"]
        assert ids(transform.rejected_records(trades)) == ["c", "d", "e"]  # null volume: removed

    def test_missing_column_raises(self, trades):
        with pytest.raises(SchemaError, match="nope"):
            make_transform(filter_step(column="nope", operator="gt", value=0)).apply(trades)


# ------------------------------------------------------------------------- expression
def expression(function, *inputs, output="out", **params):
    return {
        "name": "e",
        "type": "expression",
        "output_column": output,
        "expression": {"function": function, "inputs": list(inputs), **params},
    }


def col(name):
    return {"column": name}


def lit(value):
    return {"literal": value}


@pytest.fixture(scope="module")
def numbers(spark: SparkSession):
    rows = [(6, 3, "  Abc ", "x"), (5, 0, None, "y"), (None, 2, "q", None), (-7, None, "z", "w")]
    return spark.createDataFrame(rows, "a int, b int, s string, t string")


class TestExpressionTransform:
    def run(self, df, step):
        return values(make_transform(step).apply(df), "out")

    def test_notional_keeps_exact_decimals(self, spark):
        df = spark.createDataFrame(
            [(Decimal("185.50"), Decimal("10000"))], "price decimal(18,6), volume decimal(18,4)"
        )

        result = make_transform(expression("multiply", col("price"), col("volume"))).apply(df)

        assert values(result, "out") == [Decimal("1855000")]
        assert isinstance(result.schema["out"].dataType, DecimalType)

    def test_arithmetic_propagates_nulls(self, numbers):
        assert self.run(numbers, expression("add", col("a"), col("b"), lit(1))) == [10, 6, None, None]
        assert self.run(numbers, expression("subtract", col("a"), col("b"))) == [3, 5, None, None]
        assert self.run(numbers, expression("multiply", col("a"), lit(2))) == [12, 10, None, -14]

    def test_divide_is_safe(self, numbers):
        out = self.run(numbers, expression("divide", col("a"), col("b")))

        assert float(out[0]) == 2.0
        assert out[1:] == [None, None, None]  # zero denominator, null numerator, null denominator
        assert self.run(numbers, expression("divide", col("a"), lit(0))) == [None] * 4

    def test_round_uses_half_up_with_scale_or_default(self, spark):
        df = spark.createDataFrame([(Decimal("2.345"),), (Decimal("2.5"),), (None,)], "v decimal(10,3)")

        assert self.run(df, expression("round", col("v"), scale=2)) == [Decimal("2.35"), Decimal("2.50"), None]
        assert self.run(df, expression("round", col("v"))) == [Decimal("2"), Decimal("3"), None]

    def test_string_and_general_functions(self, numbers):
        assert self.run(numbers, expression("upper", col("s"))) == ["  ABC ", None, "Q", "Z"]
        assert self.run(numbers, expression("trim", col("s"))) == ["Abc", None, "q", "z"]
        assert self.run(numbers, expression("concat", col("t"), lit("-"), col("s"))) == [
            "x-  Abc ", None, None, "w-z",
        ]  # fmt: skip
        assert self.run(numbers, expression("coalesce", col("a"), col("b"), lit(-1))) == [6, 5, 2, -7]

    def test_float_literals_are_exact_decimals_and_functions_nest(self, numbers, spark):
        assert self.run(numbers, expression("multiply", lit(0.1), col("b"))) == [
            Decimal("0.3"), Decimal("0.0"), Decimal("0.2"), None,
        ]  # fmt: skip
        df = spark.createDataFrame([(Decimal("10.005"), 3)], "p decimal(10,3), q int")
        nested = expression(
            "round", {"function": "multiply", "inputs": [col("p"), col("q")]}, scale=1
        )

        assert self.run(df, nested) == [Decimal("30.0")]

    def test_output_column_can_be_replaced_or_created_from_time(self, numbers):
        replaced = make_transform(expression("abs", col("a"), output="a")).apply(numbers)
        stamped = make_transform(expression("current_timestamp")).apply(numbers)

        assert replaced.columns == ["a", "b", "s", "t"]
        assert values(replaced, "a") == [6, 5, None, 7]
        assert isinstance(stamped.schema["out"].dataType, TimestampType)
        assert isinstance(values(stamped, "out")[0], datetime)


# -------------------------------------------------------------------------- aggregate
def aggregate(group_by, *aggregations):
    return {
        "name": "agg",
        "type": "aggregate",
        "group_by": list(group_by),
        "aggregations": [{"column": c, "function": f, "alias": a} for c, f, a in aggregations],
    }


@pytest.fixture(scope="module")
def sales(spark: SparkSession):
    rows = [
        ("A", "d1", Decimal("10.00"), "USD"),
        ("A", "d1", Decimal("30.00"), "USD"),
        ("A", "d1", None, "USD"),
        ("A", "d2", Decimal("5.00"), "USD"),
        ("B", "d1", Decimal("7.50"), "EUR"),
        ("B", "d1", Decimal("7.50"), "EUR"),
    ]
    return spark.createDataFrame(rows, "g1 string, g2 string, v decimal(10,2), cur string")


class TestAggregateTransform:
    @staticmethod
    def rows(df, *columns):
        return sorted((tuple(row[c] for c in columns) for row in df.collect()), key=str)

    def test_sum_avg_min_max_over_groups(self, sales):
        step = aggregate(
            ["g1", "g2"], ("v", "sum", "total"), ("v", "avg", "mean"), ("v", "min", "low"), ("v", "max", "high")
        )

        result = make_transform(step).apply(sales)

        assert result.columns == ["g1", "g2", "total", "mean", "low", "high"]
        assert self.rows(result, *result.columns) == [
            ("A", "d1", Decimal("40.00"), Decimal("20"), Decimal("10.00"), Decimal("30.00")),
            ("A", "d2", Decimal("5.00"), Decimal("5"), Decimal("5.00"), Decimal("5.00")),
            ("B", "d1", Decimal("15.00"), Decimal("7.5"), Decimal("7.50"), Decimal("7.50")),
        ]

    def test_counts_and_first_on_constant_columns(self, sales):
        step = aggregate(
            ["g1"],
            ("v", "count", "non_null"),
            ("*", "count", "rows"),
            ("v", "count_distinct", "distinct_values"),
            ("cur", "first", "currency"),
        )

        result = make_transform(step).apply(sales)

        assert self.rows(result, "g1", "non_null", "rows", "distinct_values", "currency") == [
            ("A", 3, 4, 3, "USD"),
            ("B", 2, 2, 1, "EUR"),
        ]

    def test_missing_column_raises(self, sales):
        with pytest.raises(SchemaError, match="nope"):
            make_transform(aggregate(["g1"], ("nope", "sum", "total"))).apply(sales)


class TestRegistry:
    def test_every_step_type_has_a_parser_and_a_transform(self):
        assert set(STEP_PARSERS) == set(TRANSFORM_REGISTRY)
        assert all(issubclass(cls, Transform) for cls in TRANSFORM_REGISTRY.values())
