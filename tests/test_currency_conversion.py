"""
Currency conversion tests.
"""
from collections import Counter
from datetime import date
from decimal import Decimal

import pytest

from src.core.config import CurrencyConversionConfig
from src.core.exceptions import ReferenceDataError
from src.quality.currency import MISSING_FX_RATE, UNSUPPORTED_CURRENCY, CurrencyConverter

D15, D16 = date(2024, 1, 15), date(2024, 1, 16)
SCHEMA = (
    "instrument_id string, trade_date date, price decimal(18,6), volume decimal(18,4), currency string"
)
HEADER = "trade_date,from_currency,to_currency,rate"


def write_rates(tmp_path, *lines):
    path = tmp_path / "fx_rates.csv"
    path.write_text("\n".join([HEADER, *lines]) + "\n")
    return str(path)


def trades(spark, *rows):
    data = [(i, d, Decimal(p), Decimal(v), c) for i, d, p, v, c in rows]
    return spark.createDataFrame(data, SCHEMA)


def convert(spark, rates_path, df):
    config = CurrencyConversionConfig(target_currency="USD", rates_path=rates_path)
    return CurrencyConverter(spark, config, ("yyyy-MM-dd", "MM/dd/yyyy")).convert(df)


class TestConversion:
    def test_eur_is_converted_and_usd_is_untouched(self, spark, tmp_path):
        rates = write_rates(tmp_path, "2024-01-15,EUR,USD,1.10")
        df = trades(
            spark,
            ("GOOGL", D15, "120.50", "3000", "EUR"),
            ("GOOGL", D15, "142.25", "5000", "USD"),
        )

        result = convert(spark, rates, df)

        out = {r["price"]: r for r in result.valid.collect()}
        assert set(out) == {Decimal("132.55"), Decimal("142.25")}  # EUR 120.50 * 1.10
        assert {r["currency"] for r in out.values()} == {"USD"}
        assert out[Decimal("132.55")]["volume"] == Decimal("3000")  # other columns preserved
        assert result.valid.columns == df.columns
        assert result.rejected.count() == 0

    def test_rate_is_selected_by_trade_date_and_rounded_half_up_to_the_price_scale(self, spark, tmp_path):
        rates = write_rates(tmp_path, "2024-01-15,EUR,USD,1.10000000", "2024-01-16,EUR,USD,1.12345678")
        df = trades(spark, ("A", D15, "100.123457", "1", "EUR"), ("A", D16, "100", "1", "EUR"))

        out = {r["trade_date"]: r["price"] for r in convert(spark, rates, df).valid.collect()}

        assert out == {D15: Decimal("110.135803"), D16: Decimal("112.345678")}  # 110.1358027 rounds up


class TestQuarantine:
    def test_missing_unsupported_and_null_currencies_are_rejected_with_a_reason(self, spark, tmp_path):
        rates = write_rates(tmp_path, "2024-01-15,EUR,USD,1.10")
        df = trades(
            spark,
            ("A", D15, "100", "1", "USD"),
            ("B", D16, "100", "1", "EUR"),  # EUR is known, but not on this date
            ("C", D15, "100", "1", "JPY"),  # no rates at all
            ("D", D15, "100", "1", None),
            ("E", D15, "100", "1", "EUR"),
        )

        result = convert(spark, rates, df)

        assert sorted(r["instrument_id"] for r in result.valid.collect()) == ["A", "E"]
        assert Counter((r["code"], r["detail"]) for r in result.rejected.collect()) == {
            (MISSING_FX_RATE, "EUR->USD"): 1,
            (UNSUPPORTED_CURRENCY, "JPY"): 1,
            (UNSUPPORTED_CURRENCY, "<null>"): 1,
        }


class TestReferenceDataValidation:
    @pytest.fixture
    def df(self, spark):
        return trades(spark, ("A", D15, "100", "1", "EUR"))

    def test_duplicate_rates_are_fatal_even_when_formatted_differently(self, spark, tmp_path, df):
        rates = write_rates(tmp_path, "2024-01-15,EUR,USD,1.10", "01/15/2024, eur ,USD,1.11")

        with pytest.raises(ReferenceDataError, match="several rates"):
            convert(spark, rates, df)

    @pytest.mark.parametrize("line", ["2024-01-15,EUR,USD,0", ",EUR,USD,1.1"])
    def test_non_positive_or_incomplete_rate_rows_are_fatal(self, spark, tmp_path, df, line):
        with pytest.raises(ReferenceDataError, match="invalid rows"):
            convert(spark, write_rates(tmp_path, line), df)

    def test_missing_rates_file_or_wrong_header_is_fatal(self, spark, tmp_path, df):
        wrong_header = tmp_path / "rates.csv"
        wrong_header.write_text("date,from,to,rate\n2024-01-15,EUR,USD,1.1\n")

        with pytest.raises(ReferenceDataError, match="Could not read FX rates"):
            convert(spark, str(tmp_path / "nope.csv"), df)
        with pytest.raises(ReferenceDataError):
            convert(spark, str(wrong_header), df)
