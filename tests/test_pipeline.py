"""
End-to-end pipeline tests: assessment configuration, alternative configurations, logging
and CSV output.
"""
import csv
import dataclasses
import logging
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest
import yaml
from pyspark.sql.types import DateType, DecimalType, TimestampType

from src.core import PipelineConfig, PipelineEngine
from src.core.exceptions import SchemaError, SourceReadError
from src.output.writer import OutputWriter

YAML_PATH = "config/pipeline.yaml"
D15, D16 = date(2024, 1, 15), date(2024, 1, 16)


# ------------------------------------------------------------------------- helpers
class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.records: list[tuple[str, str]] = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))

    @property
    def messages(self) -> list[str]:
        return [message for _, message in self.records]


@contextmanager
def capture_logs():
    """Collect (level, message) pairs logged by the pipeline modules."""
    handler, logger = _Capture(), logging.getLogger("src")
    previous = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def raw_yaml() -> dict:
    with open(YAML_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_engine(spark, raw: dict):
    """Run the pipeline for a raw configuration; returns (result DataFrame, captured logs)."""
    with capture_logs() as logs:
        result = PipelineEngine(spark, PipelineConfig.from_dict(raw)).run()
    return result, logs


@pytest.fixture(scope="module")
def run(spark, tmp_path_factory):
    """Run the assessment pipeline once and write its CSV to a temporary directory."""
    directory = tmp_path_factory.mktemp("pipeline_output")
    config = PipelineConfig.from_yaml(YAML_PATH)
    config = dataclasses.replace(
        config, output=dataclasses.replace(config.output, path=str(directory / "result.csv"))
    )
    with capture_logs() as logs:
        result = PipelineEngine(spark, config).run()
        OutputWriter(config.output).write(result)
    return SimpleNamespace(
        config=config,
        result=result,
        directory=directory,
        logs=logs,
        rows={(r["instrument_id"], r["trade_date"]): r for r in result.collect()},
    )


# ------------------------------------------------------------ assessment configuration
class TestAssessmentPipeline:
    def test_one_row_per_instrument_and_date_with_the_configured_columns(self, run):
        assert run.result.columns == list(run.config.output.columns)
        assert len(run.rows) == run.result.count() == 9
        assert {key[0] for key in run.rows} == {"AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "NFLX"}
        assert {r["currency"] for r in run.rows.values()} == {"USD"}
        assert all(r["vwap"] is not None and r["total_volume"] > 0 for r in run.rows.values())

        schema = run.result.schema
        assert all(
            isinstance(schema[c].dataType, DecimalType)
            for c in ("total_notional", "avg_price", "total_volume", "vwap")
        )
        assert isinstance(schema["trade_date"].dataType, DateType)
        assert isinstance(schema["processed_at"].dataType, TimestampType)

    def test_aapl_financials_include_the_trimmed_identifier_and_exclude_bad_rows(self, run):
        day1, day2 = run.rows[("AAPL", D15)], run.rows[("AAPL", D16)]

        # 185.50*10000 + 186*15000 + 185.75*8000 + 185*5000 (the " AAPL" row); the row
        # with a null price is rejected
        assert day1["total_notional"] == Decimal("7056000")
        assert day1["total_volume"] == Decimal("38000")
        assert day1["avg_price"] == Decimal("185.5625")
        assert day1["vwap"] == Decimal("185.684211")
        assert (day2["total_volume"], day2["vwap"]) == (Decimal("20000"), Decimal("186.775"))

    def test_googl_eur_trades_are_converted_to_usd(self, run):
        row = run.rows[("GOOGL", D15)]

        # USD: 142.25*5000 + 142.50*7500; EUR at 1.10: 132.55*3000 + 133.10*2000
        assert row["total_notional"] == Decimal("2443850")
        assert row["total_volume"] == Decimal("17500")
        assert row["avg_price"] == Decimal("137.6")
        assert row["vwap"] == Decimal("139.648571")
        assert (run.rows[("GOOGL", D16)]["vwap"]) == Decimal("143")

    def test_invalid_filtered_and_duplicate_rows_do_not_leak_into_other_groups(self, run):
        assert ("AMZN", D15) not in run.rows  # only zero-volume trades
        assert run.rows[("MSFT", D15)]["vwap"] == Decimal("389.8")  # zero-price trade filtered
        assert run.rows[("TSLA", D15)]["vwap"] == Decimal("215")  # negative-price trade filtered
        assert run.rows[("META", D15)]["total_volume"] == Decimal("8000")  # duplicate counted once
        assert run.rows[("NVDA", D15)]["vwap"] == Decimal("548.4875")
        assert ("NFLX", D15) in run.rows  # 01/15/2024 normalised

    def test_statistics_are_logged_without_record_contents(self, spark, run):
        messages = run.logs.messages
        levels = {m.split(":")[0]: level for level, m in run.logs.records if ": total=" in m}

        assert "Pipeline started: name=instrument_daily_metrics version=1.0" in messages
        assert "Input records: total=26" in messages
        assert "Invalid records: total=4 by_reason=[NOT_POSITIVE(volume)=3, NULL_VALUE(price)=1]" in messages
        assert "Duplicate records removed: total=1 by_reason=[DUPLICATE_RECORD=1]" in messages
        assert "Currency conversion failures: total=0" in messages
        assert "Accepted input records: total=21" in messages
        assert "Filtered records: total=2 by_reason=[FILTERED_BY_FILTER_VALID_RECORDS=2]" in messages
        assert "Output validation failures: total=0" in messages
        assert "Processed records: total=9" in messages
        assert (levels["Invalid records"], levels["Currency conversion failures"]) == ("WARNING", "INFO")
        statistics = [m for m in messages if not m.startswith("Wrote")]  # skip the temp path
        assert not any("AAPL" in m or "185.5" in m for m in statistics)  # no record contents
        # The engine cache is released before run() returns.
        assert spark.sparkContext._jsc.sc().getPersistentRDDs().isEmpty()

    def test_csv_has_the_processed_rows_only_and_repeated_runs_replace_it(self, spark, run):
        target = run.directory / "result.csv"

        for _ in range(2):  # run again: the previous CSV is replaced, not appended to
            OutputWriter(run.config.output).write(PipelineEngine(spark, run.config).run())

        with open(target, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        aapl = next(r for r in rows if r["instrument_id"] == "AAPL" and r["trade_date"] == "2024-01-15")
        assert list(rows[0]) == list(run.config.output.columns)
        assert len(rows) == 9
        assert "AMZN" not in {r["instrument_id"] for r in rows}
        assert (Decimal(aapl["total_notional"]), Decimal(aapl["vwap"])) == (Decimal("7056000"), Decimal("185.684211"))
        assert [p.name for p in run.directory.iterdir()] == ["result.csv"]


# -------------------------------------------------------- other valid configurations
class TestGenericEngine:
    """The engine runs any valid YAML, not just the assessment configuration."""

    def test_alternative_pipeline(self, spark):
        raw = raw_yaml()
        del raw["currency_conversion"], raw["validation"]
        raw["transformations"] = [
            {
                "name": "valid_trades",
                "type": "filter",
                "condition": {
                    "operator": "and",
                    "conditions": [
                        {"column": "price", "operator": "gt", "value": 0},
                        {"column": "volume", "operator": "gt", "value": 0},
                    ],
                },
            },
            {
                "name": "by_currency",
                "type": "aggregate",
                "group_by": ["currency"],
                "aggregations": [
                    {"column": "*", "function": "count", "alias": "trades"},
                    {"column": "price", "function": "min", "alias": "min_price"},
                    {"column": "price", "function": "max", "alias": "max_price"},
                    {"column": "instrument_id", "function": "count_distinct", "alias": "instruments"},
                ],
            },
            {
                "name": "round_max",
                "type": "expression",
                "output_column": "max_price_rounded",
                "expression": {"function": "round", "scale": 0, "inputs": [{"column": "max_price"}]},
            },
        ]
        raw["output"]["columns"] = [
            "currency", "trades", "min_price", "max_price", "instruments", "max_price_rounded",
        ]  # fmt: skip

        result, _ = run_engine(spark, raw)

        rows = {r["currency"]: r for r in result.collect()}
        assert set(rows) == {"USD", "EUR"}  # no FX conversion configured
        usd, eur = rows["USD"], rows["EUR"]
        assert (usd["trades"], usd["instruments"]) == (17, 7)
        assert (usd["min_price"], usd["max_price"], usd["max_price_rounded"]) == (
            Decimal("142.25"), Decimal("549.5"), Decimal("550"),
        )  # fmt: skip
        assert (eur["trades"], eur["instruments"], eur["max_price_rounded"]) == (2, 1, Decimal("121"))

    def test_undefined_vwap_is_excluded_instead_of_failing(self, spark):
        raw = raw_yaml()
        # Without the input volume rule the zero-volume AMZN trades reach the division.
        raw["data_quality"]["input_rules"] = [
            r for r in raw["data_quality"]["input_rules"] if r["column"] != "volume"
        ]

        result, logs = run_engine(spark, raw)

        assert result.count() == 9
        assert "AMZN" not in {r["instrument_id"] for r in result.collect()}
        # Division by the zero total volume gives a null VWAP, the first failing output rule.
        assert "Output validation failures: total=1 by_reason=[NULL_VALUE(vwap)=1]" in logs.messages
        assert "Processed records: total=9" in logs.messages

    def test_missing_fx_rate_excludes_the_trades_and_logs_the_failure(self, spark, tmp_path):
        raw = raw_yaml()
        rates = tmp_path / "rates.csv"
        rates.write_text("trade_date,from_currency,to_currency,rate\n2024-01-16,EUR,USD,1.10\n")
        raw["currency_conversion"]["rates_path"] = str(rates)

        result, logs = run_engine(spark, raw)

        assert "Currency conversion failures: total=2 by_reason=[MISSING_FX_RATE(EUR->USD)=2]" in logs.messages
        assert "Accepted input records: total=19" in logs.messages  # 21 minus the 2 EUR trades
        googl = {(r["instrument_id"], r["trade_date"]): r for r in result.collect()}
        assert googl[("GOOGL", D15)]["total_volume"] == Decimal("12500")  # USD trades only


# -------------------------------------------------------------------- fatal errors
def _missing_source(raw):
    raw["source"]["path"] = "data/input/does_not_exist.csv"


def _wrong_header(raw):  # same columns, different order: the file header no longer matches
    columns = raw["source"]["columns"]
    raw["source"]["columns"] = {k: columns[k] for k in ("instrument_id", "trade_date", "volume", "price", "currency")}


def _unknown_output_column(raw):
    raw["output"]["columns"] = [*raw["output"]["columns"], "not_computed"]


class TestFatalErrors:
    @pytest.mark.parametrize(
        "mutate, error, message",
        [
            (_missing_source, SourceReadError, "does_not_exist"),
            (_wrong_header, SourceReadError, "does not match"),
            (_unknown_output_column, SchemaError, "not_computed"),
        ],
    )
    def test_fatal_problems_raise_a_pipeline_error_and_release_the_cache(self, spark, mutate, error, message):
        raw = raw_yaml()
        mutate(raw)

        with pytest.raises(error, match=message):
            run_engine(spark, raw)

        assert spark.sparkContext._jsc.sc().getPersistentRDDs().isEmpty()
