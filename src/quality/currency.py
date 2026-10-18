"""
Currency conversion to a single target currency using reference exchange rates.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from src.core.exceptions import ReferenceDataError
from src.quality.normalizer import blank_to_null, parse_date, parse_decimal
from src.quality.rejections import (
    ERRORS_COLUMN,
    SplitResult,
    Stage,
    collect_errors,
    error,
    split_on_errors,
)
from src.transforms.base import require_columns

if TYPE_CHECKING:
    from src.core.config import CurrencyConversionConfig

logger = logging.getLogger(__name__)

UNSUPPORTED_CURRENCY = "UNSUPPORTED_CURRENCY"
MISSING_FX_RATE = "MISSING_FX_RATE"

_RATES_SCHEMA = StructType(
    [
        StructField(name, StringType(), True)
        for name in ("trade_date", "from_currency", "to_currency", "rate")
    ]
)
_RATE_PRECISION, _RATE_SCALE = 18, 8
_FX_DATE, _FX_CURRENCY, _FX_RATE, _FX_KNOWN = "_fx_date", "_fx_currency", "_fx_rate", "_fx_known"
_FX_COLUMNS = (_FX_DATE, _FX_CURRENCY, _FX_RATE, _FX_KNOWN)
_MAX_REPORTED_PROBLEMS = 5


class CurrencyConverter:
    """
    Converts prices to ``target_currency`` with a date-specific rate table.

    Rates come from a CSV (``trade_date, from_currency, to_currency, rate``). Records
    already in the target currency are unchanged; others are joined to the rate for
    their trade date and currency and multiplied, rounded half-up to the price's own
    decimal scale. The currency column is set to the target currency and every other
    column (including volume) is preserved.

    Reference data problems are fatal (``ReferenceDataError``): non-positive or
    unparseable rates, or more than one rate for a date/currency pair, are never
    resolved by picking one arbitrarily. Record-level problems are quarantined:
    ``UNSUPPORTED_CURRENCY`` (null currency, or one that has no rate on any date) and
    ``MISSING_FX_RATE`` (a supported currency with no rate for that trade date).
    """

    def __init__(
        self,
        spark: SparkSession,
        config: "CurrencyConversionConfig",
        date_formats: Sequence[str],
    ):
        self._spark = spark
        self._config = config
        self._date_formats = tuple(date_formats)

    def convert(self, df: DataFrame) -> SplitResult:
        config = self._config
        require_columns(
            df,
            [config.price_column, config.currency_column, config.date_column],
            "currency_conversion",
        )
        rates = self._load_rates()
        known = rates.select(F.col(_FX_CURRENCY).alias(_FX_KNOWN)).distinct()

        currency = F.col(config.currency_column)
        joined = df.join(
            F.broadcast(rates),
            (F.col(config.date_column) == F.col(_FX_DATE)) & (currency == F.col(_FX_CURRENCY)),
            "left",
        ).join(F.broadcast(known), currency == F.col(_FX_KNOWN), "left")

        target = F.lit(config.target_currency)
        is_target = currency == target
        unsupported = currency.isNull() | (~is_target & F.col(_FX_KNOWN).isNull())
        missing_rate = ~is_target & F.col(_FX_KNOWN).isNotNull() & F.col(_FX_RATE).isNull()
        checked = joined.withColumn(
            ERRORS_COLUMN,
            collect_errors(
                # Details are currency codes only: low cardinality, no financial values.
                F.when(
                    unsupported, error(UNSUPPORTED_CURRENCY, F.coalesce(currency, F.lit("<null>")))
                ),
                F.when(
                    missing_rate,
                    error(MISSING_FX_RATE, F.concat(currency, F.lit(f"->{config.target_currency}"))),
                ),
            ),
        )
        split = split_on_errors(checked, Stage.CURRENCY_CONVERSION)

        price_type = df.schema[config.price_column].dataType
        converted_price = F.when(is_target, F.col(config.price_column)).otherwise(
            (F.col(config.price_column) * F.col(_FX_RATE)).cast(price_type)
        )
        converted = (
            split.valid.withColumn(config.price_column, converted_price)
            .withColumn(config.currency_column, target)
            .drop(*_FX_COLUMNS)
        )
        return SplitResult(valid=converted, rejected=split.rejected)

    def _load_rates(self) -> DataFrame:
        """Load, validate and return the rates to the target currency (``_fx_*`` columns)."""
        config = self._config
        self._spark.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        try:
            raw = (
                self._spark.read.option("header", True)
                .option("enforceSchema", False)
                .schema(_RATES_SCHEMA)
                .csv(config.rates_path)
            )
            parsed = raw.select(
                parse_date(blank_to_null(F.trim("trade_date")), self._date_formats).alias(_FX_DATE),
                F.upper(blank_to_null(F.trim("from_currency"))).alias(_FX_CURRENCY),
                F.upper(blank_to_null(F.trim("to_currency"))).alias("_fx_to"),
                parse_decimal(
                    blank_to_null(F.trim("rate")), _RATE_PRECISION, _RATE_SCALE
                ).alias(_FX_RATE),
                F.to_json(F.struct(*[F.col(c) for c in raw.columns])).alias("_fx_raw"),
            )
            invalid = (
                F.col(_FX_DATE).isNull()
                | F.col(_FX_CURRENCY).isNull()
                | F.col("_fx_to").isNull()
                | F.col(_FX_RATE).isNull()
                | (F.col(_FX_RATE) <= 0)
            )
            bad_rows = [
                row["_fx_raw"]
                for row in parsed.filter(invalid).limit(_MAX_REPORTED_PROBLEMS).collect()
            ]
            duplicate_keys = [
                f"{row[_FX_DATE]} {row[_FX_CURRENCY]}->{row['_fx_to']} ({row['count']} rates)"
                for row in parsed.groupBy(_FX_DATE, _FX_CURRENCY, "_fx_to")
                .count()
                .filter(F.col("count") > 1)
                .limit(_MAX_REPORTED_PROBLEMS)
                .collect()
            ]
        except Exception as exc:
            raise ReferenceDataError(
                f"Could not read FX rates '{config.rates_path}': {exc}"
            ) from exc

        if bad_rows:
            raise ReferenceDataError(
                f"FX rates '{config.rates_path}' contain invalid rows (date, currencies and a "
                f"positive numeric rate are required), e.g. {bad_rows}"
            )
        if duplicate_keys:
            raise ReferenceDataError(
                f"FX rates '{config.rates_path}' define several rates for the same "
                f"date/currency pair: {duplicate_keys}"
            )
        logger.info("Loaded FX rates from %s", config.rates_path)
        return parsed.filter(F.col("_fx_to") == config.target_currency).select(
            _FX_DATE, _FX_CURRENCY, _FX_RATE
        )
