"""
Data normalization: whitespace, casing, dates and numeric types.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

from src.quality.rejections import ERRORS_COLUMN, collect_errors, error
from src.transforms.base import require_columns

if TYPE_CHECKING:
    from src.core.config import ColumnSpec, NormalizationConfig

INVALID_DATE = "INVALID_DATE"
INVALID_NUMERIC = "INVALID_NUMERIC"


def blank_to_null(column: Column) -> Column:
    """Treat empty and whitespace-only strings as null."""
    return F.when(F.trim(column) == "", F.lit(None)).otherwise(column)


def parse_date(column: Column, formats: Sequence[str]) -> Column:
    """Parse a date string with the first matching format (null if none match)."""
    return F.coalesce(*[F.to_date(column, fmt) for fmt in formats])


def parse_decimal(column: Column, precision: int, scale: int) -> Column:
    """Cast a string to a decimal (null if unparseable or out of range)."""
    return column.cast(DecimalType(precision, scale))


class DataNormalizer:
    """
    Converts raw string records to typed, standardised values.

    Per column type:
      * ``string``: blank -> null; configured columns are trimmed, inner whitespace is
        collapsed, and (for ``uppercase_columns``) upper-cased.
      * ``date``: parsed with the configured formats; first match wins, so ambiguous
        formats (``MM/dd`` vs ``dd/MM``) resolve in the listed order.
      * ``decimal(p,s)``: exact decimal; values with more fractional digits are rounded
        half-up to the scale.

    Values that are present but cannot be parsed become null and add an
    ``INVALID_DATE`` / ``INVALID_NUMERIC`` error (detail: the column name) to ``_errors``
    so the record can be quarantined. Missing values are *not* errors here; null checks
    belong to the input rules.
    """

    def __init__(self, columns: Sequence["ColumnSpec"], config: "NormalizationConfig"):
        self._columns = tuple(columns)
        self._config = config
        string_columns = [c.name for c in self._columns if c.type_name == "string"]
        trim = config.trim_columns
        self._trim = set(string_columns if trim is None else trim)
        self._uppercase = set(config.uppercase_columns)

    def normalize(self, raw: DataFrame) -> DataFrame:
        """Return the typed columns followed by ``_errors``."""
        # Unparseable dates must become null (and be quarantined), not fail the job.
        raw.sparkSession.conf.set("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        require_columns(raw, [column.name for column in self._columns], "source data")

        values: list[Column] = []
        failures: list[Column] = []
        for spec in self._columns:
            value, failure = self._normalize_column(spec)
            values.append(value.alias(spec.name))
            if failure is not None:
                failures.append(failure)
        return raw.select(*values, collect_errors(*failures).alias(ERRORS_COLUMN))

    def _normalize_column(self, spec: "ColumnSpec") -> tuple[Column, Optional[Column]]:
        column = F.col(spec.name)
        if spec.type_name == "string":
            value = blank_to_null(column)
            if spec.name in self._trim:
                value = F.trim(F.regexp_replace(value, r"\s+", " "))
            if spec.name in self._uppercase:
                value = F.upper(value)
            return value, None

        text = blank_to_null(F.trim(column))
        if spec.type_name == "date":
            value = parse_date(text, self._config.date_formats)
            code = INVALID_DATE
        else:
            value = parse_decimal(text, spec.precision, spec.scale)
            code = INVALID_NUMERIC
        return value, F.when(text.isNotNull() & value.isNull(), error(code, spec.name))
