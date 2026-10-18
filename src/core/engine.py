"""
Pipeline execution engine.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from functools import reduce

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F

from src.core.config import ColumnSpec, PipelineConfig
from src.core.reader import SourceReader
from src.quality.currency import CurrencyConverter
from src.quality.normalizer import DataNormalizer
from src.quality.rejections import (
    DUPLICATE_RECORD,
    Stage,
    rejected_frame,
    split_on_errors,
)
from src.quality.validator import DataQualityValidator, RuleValidator
from src.transforms.base import require_columns
from src.transforms.registry import build_transform

logger = logging.getLogger(__name__)

_PROCESSED = "processed"  # pseudo-stage label for the final record count


class PipelineEngine:
    """
    Executes a pipeline configuration.

    Stages: (1) read source, (2) normalise, (3) input data quality and currency
    conversion, (4) YAML transformations in order, (5) output validation. (6) Writing the
    result is left to ``OutputWriter``, so ``run()`` writes no files.

    Record-level problems never fail the run: those records are excluded from the result
    and counted by reason in the log (never logged individually). Fatal problems raise a
    ``PipelineError``.

    ``run()`` returns a lazy DataFrame. It triggers two small Spark jobs for the log
    statistics: one materialises the normalised data (cached because every stage's
    valid/rejected branches reuse it) and one aggregates all counts together. The cache
    is released before ``run()`` returns.
    """

    def __init__(self, spark: SparkSession, config: PipelineConfig):
        self.spark = spark
        self.config = config

    def run(self) -> DataFrame:
        """Execute the pipeline and return the processed (valid) records."""
        logger.info("Pipeline started: name=%s version=%s", self.config.name, self.config.version)
        raw = SourceReader(self.spark, self.config.source).read()
        normalized = self._normalize(raw).persist(StorageLevel.MEMORY_AND_DISK)
        try:
            total_input = normalized.count()  # also fills the cache
            split = split_on_errors(normalized, Stage.NORMALIZATION)
            rejected = [split.rejected]
            df = self._check_input_quality(split.valid, rejected)
            df = self._convert_currency(df, rejected)
            df = self._transform(df, rejected)
            processed = self._validate_output(df, rejected)
            self._log_summary(total_input, self._count_by_reason(rejected, processed))
        finally:
            normalized.unpersist()
        return processed

    # ---------------------------------------------------------------------- stages
    def _normalize(self, raw: DataFrame) -> DataFrame:
        columns = self.config.source.columns or tuple(ColumnSpec(name) for name in raw.columns)
        return DataNormalizer(columns, self.config.data_quality.normalization).normalize(raw)

    def _check_input_quality(self, df: DataFrame, rejected: list[DataFrame]) -> DataFrame:
        split = DataQualityValidator(self.config.data_quality).validate(df)
        rejected.append(split.rejected)
        return split.valid

    def _convert_currency(self, df: DataFrame, rejected: list[DataFrame]) -> DataFrame:
        conversion = self.config.currency_conversion
        if conversion is None:
            return df
        converter = CurrencyConverter(
            self.spark, conversion, self.config.data_quality.normalization.date_formats
        )
        split = converter.convert(df)
        rejected.append(split.rejected)
        return split.valid

    def _transform(self, df: DataFrame, rejected: list[DataFrame]) -> DataFrame:
        for step in self.config.transformations:
            transform = build_transform(step)
            removed = transform.rejected_records(df)
            if removed is not None:
                rejected.append(
                    rejected_frame(removed, Stage.TRANSFORMATION_FILTER, f"FILTERED_BY_{step.name.upper()}")
                )
            df = transform.apply(df)
            logger.info("Applied transformation: name=%s type=%s", step.name, step.type)
        return df

    def _validate_output(self, df: DataFrame, rejected: list[DataFrame]) -> DataFrame:
        columns = list(self.config.output.columns)
        require_columns(df, columns, "output.columns")
        split = RuleValidator(self.config.validation_rules, Stage.OUTPUT_VALIDATION).validate(df)
        rejected.append(split.rejected)
        return split.valid.select(*columns)

    # --------------------------------------------------------------------- logging
    @staticmethod
    def _count_by_reason(rejected: list[DataFrame], processed: DataFrame) -> list[Row]:
        """
        One Spark job returning ``stage, code, detail, n`` rows: rejections grouped by
        reason plus the processed record count. The result has one row per distinct
        reason, so collecting it to the driver is cheap at any data volume.
        """
        by_reason = reduce(DataFrame.unionByName, rejected).groupBy("stage", "code", "detail").agg(
            F.count(F.lit(1)).alias("n")
        )
        total = processed.agg(F.count(F.lit(1)).alias("n")).select(
            F.lit(_PROCESSED).alias("stage"), F.lit("").alias("code"), F.lit("").alias("detail"), "n"
        )
        return by_reason.unionByName(total).collect()

    @staticmethod
    def _log_summary(total_input: int, rows: list[Row]) -> None:
        """Log the data quality statistics: input, rejections by reason, duplicates, output."""
        reasons: dict[str, dict[str, int]] = defaultdict(dict)
        processed = 0
        for row in rows:
            if row["stage"] == _PROCESSED:
                processed = row["n"]
            else:
                label = f"{row['code']}({row['detail']})" if row["detail"] else row["code"]
                reasons[row["stage"]][label] = row["n"]

        input_stage = reasons[Stage.INPUT_DATA_QUALITY.value]
        duplicates = input_stage.pop(DUPLICATE_RECORD, 0)
        invalid: dict[str, int] = defaultdict(int)
        for stage in (Stage.NORMALIZATION, Stage.INPUT_DATA_QUALITY):
            for label, count in reasons[stage.value].items():
                invalid[label] += count
        conversion = reasons[Stage.CURRENCY_CONVERSION.value]
        accepted = total_input - sum(invalid.values()) - duplicates - sum(conversion.values())

        logger.info("Input records: total=%d", total_input)
        _log_count("Invalid records", invalid)
        _log_count("Duplicate records removed", {DUPLICATE_RECORD: duplicates} if duplicates else {})
        _log_count("Currency conversion failures", conversion)
        logger.info("Accepted input records: total=%d", accepted)
        _log_count("Filtered records", reasons[Stage.TRANSFORMATION_FILTER.value])
        _log_count("Output validation failures", reasons[Stage.OUTPUT_VALIDATION.value])
        logger.info("Processed records: total=%d", processed)


def _log_count(label: str, by_reason: dict[str, int]) -> None:
    """Log ``label: total=N by_reason=[...]``; a warning when anything was rejected."""
    total = sum(by_reason.values())
    breakdown = ", ".join(f"{reason}={count}" for reason, count in sorted(by_reason.items()))
    logger.log(
        logging.WARNING if total else logging.INFO,
        "%s: total=%d%s",
        label,
        total,
        f" by_reason=[{breakdown}]" if total else "",
    )
