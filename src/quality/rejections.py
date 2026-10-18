"""
Quarantine helpers shared by every pipeline stage.

A stage that finds bad records attaches an array of ``{code, detail}`` errors to each row
(``_errors``) and splits the DataFrame into valid and rejected parts. The rejected part
is reduced to ``stage, code, detail`` (no record contents), so the engine can count
rejections by reason with one aggregation. ``detail`` is a static, value-free label such
as the column name, never the offending data.

Internal columns are prefixed with ``_`` and never reach the processed output.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

ERRORS_COLUMN = "_errors"
_ERRORS_DDL = "array<struct<code:string,detail:string>>"

DUPLICATE_RECORD = "DUPLICATE_RECORD"


class Stage(str, Enum):
    """Pipeline stage that rejected a record."""

    NORMALIZATION = "normalization"
    INPUT_DATA_QUALITY = "input_data_quality"
    CURRENCY_CONVERSION = "currency_conversion"
    TRANSFORMATION_FILTER = "transformation_filter"
    OUTPUT_VALIDATION = "output_validation"


@dataclass(frozen=True)
class SplitResult:
    """A DataFrame split into records that continue and records that are quarantined."""

    valid: DataFrame
    rejected: DataFrame


def error(code: str, detail: Column | str = "") -> Column:
    """Build one ``{code, detail}`` error struct."""
    text = F.lit(detail) if isinstance(detail, str) else detail
    return F.struct(F.lit(code).alias("code"), text.alias("detail"))


def collect_errors(*conditional_errors: Column) -> Column:
    """Combine ``when(condition, error(...))`` columns into one array without the nulls."""
    if not conditional_errors:
        return F.array().cast(_ERRORS_DDL)
    return F.filter(F.array(*conditional_errors), lambda e: e.isNotNull()).cast(_ERRORS_DDL)


def rejected_frame(
    df: DataFrame, stage: Stage, code: Column | str, detail: Column | str = ""
) -> DataFrame:
    """Describe ``df`` rows as rejected records: one ``stage, code, detail`` row each."""
    code_column = F.lit(code) if isinstance(code, str) else code
    detail_column = F.lit(detail) if isinstance(detail, str) else detail
    return df.select(
        F.lit(stage.value).alias("stage"),
        code_column.alias("code"),
        detail_column.alias("detail"),
    )


def split_on_errors(df: DataFrame, stage: Stage) -> SplitResult:
    """
    Split ``df`` on its ``_errors`` column.

    Rows without errors are returned (minus ``_errors``). Rows with errors are rejected,
    each reported once under its first error.
    """
    error_count = F.size(F.col(ERRORS_COLUMN))
    first_error = F.col(ERRORS_COLUMN).getItem(0)
    return SplitResult(
        valid=df.filter(error_count == 0).drop(ERRORS_COLUMN),
        rejected=rejected_frame(
            df.filter(error_count > 0),
            stage,
            first_error.getField("code"),
            first_error.getField("detail"),
        ),
    )
