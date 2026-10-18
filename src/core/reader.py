"""
Source reader.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from src.core.exceptions import SourceReadError

if TYPE_CHECKING:
    from src.core.config import SourceConfig

logger = logging.getLogger(__name__)


class SourceReader:
    """
    Reads the configured source without losing information.

    Every column is read as a string using the explicit schema from ``source.columns``
    (or the file header when none is declared), with no whitespace stripping and no
    value conversion, so invalid raw values survive for validation. The DataFrame is
    lazy: nothing is scanned here except the header line.
    """

    def __init__(self, spark: SparkSession, config: "SourceConfig"):
        self._spark = spark
        self._config = config

    def read(self) -> DataFrame:
        path = self._config.path
        reader = (
            self._spark.read.option("header", True)
            .option("mode", "PERMISSIVE")
            .option("ignoreLeadingWhiteSpace", False)
            .option("ignoreTrailingWhiteSpace", False)
        )
        try:
            if not self._config.columns:
                return reader.csv(path)
            self._check_header(path)
            schema = StructType(
                [StructField(column.name, StringType(), True) for column in self._config.columns]
            )
            return reader.schema(schema).csv(path)
        except SourceReadError:
            raise
        except Exception as exc:  # Spark/JVM errors surface as many exception types
            raise SourceReadError(f"Could not read source '{path}': {exc}") from exc

    def _check_header(self, path: str) -> None:
        """Fail early if the file header does not match ``source.columns`` (by position)."""
        header = self._spark.read.option("header", True).csv(path).columns
        expected = [column.name for column in self._config.columns]
        if [name.lower() for name in header] != [name.lower() for name in expected]:
            raise SourceReadError(
                f"Source '{path}' header {header} does not match source.columns {expected}"
            )
