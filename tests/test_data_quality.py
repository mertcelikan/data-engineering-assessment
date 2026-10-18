"""
Normalization, input rule validation and duplicate detection tests.
"""
from collections import Counter
from datetime import date
from decimal import Decimal

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, DecimalType

from src.core.config import ColumnSpec, DataQualityConfig, NormalizationConfig, RuleSpec
from src.core.exceptions import SchemaError
from src.quality.normalizer import DataNormalizer
from src.quality.rejections import DUPLICATE_RECORD, Stage, split_on_errors
from src.quality.validator import DataQualityValidator, RuleValidator

SPECS = (
    ColumnSpec("instrument_id"),
    ColumnSpec("trade_date", "date"),
    ColumnSpec("price", "decimal", 18, 6),
    ColumnSpec("volume", "decimal", 18, 4),
    ColumnSpec("currency"),
)
NORMALIZATION = NormalizationConfig(
    trim_columns=("instrument_id", "currency"),
    uppercase_columns=("currency",),
    date_formats=("yyyy-MM-dd", "MM/dd/yyyy"),
)
RAW_SCHEMA = "instrument_id string, trade_date string, price string, volume string, currency string"
GOOD = ("AAPL", "2024-01-15", "185.50", "10000", "USD")
INPUT_RULES = (
    RuleSpec("instrument_id", "not_null"),
    RuleSpec("trade_date", "not_null"),
    RuleSpec("price", "not_null"),
    RuleSpec("volume", "positive"),
)


def normalize(spark: SparkSession, *records):
    return DataNormalizer(SPECS, NORMALIZATION).normalize(spark.createDataFrame(list(records), RAW_SCHEMA))


def input_quality(spark: SparkSession, *records, rules=INPUT_RULES):
    """Normalise, then run the input rules and duplicate detection (as the engine does)."""
    normalized = split_on_errors(normalize(spark, *records), Stage.NORMALIZATION)
    config = DataQualityConfig(normalization=NORMALIZATION, input_rules=tuple(rules))
    return normalized, DataQualityValidator(config).validate(normalized.valid)


def reasons(df):
    """Rejected records as a multiset of (code, detail)."""
    return Counter((row["code"], row["detail"]) for row in df.collect())


class TestNormalizer:
    def test_strings_are_trimmed_collapsed_uppercased_and_blanks_become_null(self, spark):
        df = normalize(
            spark,
            ("  AAPL ", GOOD[1], "1", "1", " usd "),
            ("BRK   B", *GOOD[1:]),
            ("", "   ", "", "  ", ""),
        )

        rows = df.collect()

        assert [(r["instrument_id"], r["currency"]) for r in rows[:2]] == [("AAPL", "USD"), ("BRK B", "USD")]
        assert [rows[2][c.name] for c in SPECS] == [None] * 5
        assert rows[2]["_errors"] == []  # missing values are for the input rules, not errors

    @pytest.mark.parametrize("raw", ["2024-01-15", "01/15/2024"])
    def test_both_date_formats_give_the_same_date(self, spark, raw):
        df = normalize(spark, ("AAPL", raw, "1", "1", "USD"))

        assert df.collect()[0]["trade_date"] == date(2024, 1, 15)
        assert isinstance(df.schema["trade_date"].dataType, DateType)

    def test_decimals_are_typed_and_rounded_half_up(self, spark):
        df = normalize(spark, ("AAPL", GOOD[1], " 185.5 ", "10000", "USD"), ("AAPL", GOOD[1], "1.2345675", "1", "USD"))

        rows = df.collect()

        assert (rows[0]["price"], rows[0]["volume"]) == (Decimal("185.5"), Decimal("10000"))
        assert rows[1]["price"] == Decimal("1.234568")
        assert (df.schema["price"].dataType, df.schema["volume"].dataType) == (DecimalType(18, 6), DecimalType(18, 4))

    def test_unparseable_values_become_null_and_are_flagged_by_column(self, spark):
        df = normalize(
            spark,
            ("AAPL", "2024-13-45", "1", "1", "USD"),
            ("AAPL", GOOD[1], "12,5", "1", "USD"),
            ("AAPL", GOOD[1], "-5", "0", "USD"),  # negative/zero are valid numbers
        )

        rows = df.collect()

        assert rows[0]["trade_date"] is None and rows[1]["price"] is None
        assert [(e["code"], e["detail"]) for e in rows[0]["_errors"]] == [("INVALID_DATE", "trade_date")]
        assert [(e["code"], e["detail"]) for e in rows[1]["_errors"]] == [("INVALID_NUMERIC", "price")]
        assert rows[2]["_errors"] == [] and rows[2]["price"] == Decimal("-5")

    def test_missing_source_column_raises(self, spark):
        raw = spark.createDataFrame([GOOD], RAW_SCHEMA).drop("price")

        with pytest.raises(SchemaError, match="price"):
            DataNormalizer(SPECS, NORMALIZATION).normalize(raw)


class TestInputRules:
    def test_null_zero_and_negative_values_are_rejected_once_by_the_first_failing_rule(self, spark):
        _, result = input_quality(
            spark,
            GOOD,
            ("AAPL", GOOD[1], "", "10", "USD"),  # null price
            ("AMZN", GOOD[1], "155", "0", "USD"),  # zero volume
            ("MSFT", GOOD[1], "391", "-500", "USD"),  # negative volume
            ("MSFT", GOOD[1], "391", "", "USD"),  # null never satisfies a rule
            ("", GOOD[1], "", "-1", "USD"),  # three failures: reported once, under the first
        )

        assert result.valid.count() == 1
        assert reasons(result.rejected) == {
            ("NULL_VALUE", "price"): 1,
            ("NOT_POSITIVE", "volume"): 3,
            ("NULL_VALUE", "instrument_id"): 1,
        }

    def test_price_sign_is_a_business_filter_not_an_input_rule(self, spark):
        _, result = input_quality(spark, ("A", GOOD[1], "0", "1", "USD"), ("B", GOOD[1], "-5", "1", "USD"))

        assert result.valid.count() == 2
        assert result.rejected.count() == 0
        assert "_errors" not in result.valid.columns

    @pytest.mark.parametrize(
        "check, value, passing",
        [("gt", 5, ["10"]), ("between", [2, 6], ["5"])],
    )
    def test_generic_checks(self, spark, check, value, passing):
        records = [("A", GOOD[1], "1", v, "USD") for v in ("1", "5", "10")]

        _, result = input_quality(spark, *records, rules=[RuleSpec("volume", check, value)])

        assert sorted(str(int(r["volume"])) for r in result.valid.collect()) == passing
        assert result.valid.count() + result.rejected.count() == 3

    def test_rule_on_missing_column_raises(self, spark):
        valid = split_on_errors(normalize(spark, GOOD), Stage.NORMALIZATION).valid

        with pytest.raises(SchemaError, match="nope"):
            RuleValidator([RuleSpec("nope", "not_null")], Stage.INPUT_DATA_QUALITY).validate(valid)


class TestDuplicates:
    def test_exact_duplicates_are_removed_including_after_normalization(self, spark):
        _, result = input_quality(
            spark,
            GOOD,
            GOOD,
            (" AAPL", "01/15/2024", "185.500", "10000.0", " usd"),  # same record once normalised
            ("AAPL", GOOD[1], "186", "10000", "USD"),  # different price: a separate trade
        )

        assert result.valid.count() == 2
        assert reasons(result.rejected) == {(DUPLICATE_RECORD, ""): 2}

    def test_same_instrument_and_date_with_different_values_is_not_a_duplicate(self, spark):
        _, result = input_quality(
            spark,
            ("AAPL", GOOD[1], "185.50", "10000", "USD"),
            ("AAPL", GOOD[1], "186.00", "15000", "USD"),
            ("AAPL", GOOD[1], "186.00", "10000", "USD"),
        )

        assert result.valid.count() == 3
        assert result.rejected.count() == 0

    def test_every_record_is_accounted_for_exactly_once(self, spark):
        bad = ("AAPL", GOOD[1], "", "10", "USD")
        records = [
            GOOD,
            GOOD,  # duplicate
            bad,
            bad,  # an invalid duplicate is reported as invalid, not as a duplicate
            ("AMZN", GOOD[1], "155", "0", "USD"),
            ("NFLX", "01/15/2024", "480", "4000", "USD"),
            ("X", "bad-date", "1", "1", "USD"),  # normalization failure
        ]

        normalized, result = input_quality(spark, *records)

        assert reasons(normalized.rejected) == {("INVALID_DATE", "trade_date"): 1}
        assert reasons(result.rejected) == {
            (DUPLICATE_RECORD, ""): 1,
            ("NULL_VALUE", "price"): 2,
            ("NOT_POSITIVE", "volume"): 1,
        }
        assert result.valid.count() == 2
        assert normalized.rejected.count() + result.rejected.count() + result.valid.count() == len(records)
