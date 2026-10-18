"""
OutputWriter tests: one CSV file with a header, replaced on every run.
"""
import csv

import pytest
from pyspark.sql import functions as F

from src.core.config import OutputConfig
from src.core.exceptions import OutputWriteError
from src.output.writer import OutputWriter

SCHEMA = "id string, amount int"


def write(df, path):
    return OutputWriter(OutputConfig(columns=("id", "amount"), path=str(path))).write(df)


def read_rows(path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def files(directory) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


class TestOutputWriter:
    def test_writes_one_csv_with_a_header_whatever_the_partitioning(self, spark, tmp_path):
        target = tmp_path / "nested" / "result.csv"  # the parent directory is created
        rows = [("plain", 1), ('a,"b"', 2), (None, 3)] + [(f"r{i}", i) for i in range(100)]
        df = spark.createDataFrame(rows, SCHEMA).repartition(8)

        assert write(df, target) == target.resolve()

        written = read_rows(target)
        assert written[0] == ["id", "amount"]
        assert len(written) == 1 + len(rows)
        assert ['a,"b"', "2"] in written and ["", "3"] in written  # quoting round-trips, null -> empty
        assert files(target.parent) == ["result.csv"]  # no part files, _SUCCESS or staging leftovers

    def test_a_second_run_replaces_the_file(self, spark, tmp_path):
        target = tmp_path / "result.csv"
        write(spark.createDataFrame([("A", 1), ("B", 2), ("C", 3)], SCHEMA), target)

        write(spark.createDataFrame([("Z", 9)], SCHEMA), target)

        assert read_rows(target) == [["id", "amount"], ["Z", "9"]]
        assert files(tmp_path) == ["result.csv"]

    def test_an_empty_result_writes_a_header_only_file(self, spark, tmp_path):
        target = tmp_path / "result.csv"

        write(spark.createDataFrame([], SCHEMA), target)

        assert read_rows(target) == [["id", "amount"]]

    def test_a_failed_write_keeps_the_previous_file_and_leaves_no_leftovers(self, spark, tmp_path):
        target = tmp_path / "result.csv"
        write(spark.createDataFrame([("A", 1)], SCHEMA), target)
        failing = spark.createDataFrame([("B", 2)], SCHEMA).withColumn("amount", F.raise_error("boom"))

        with pytest.raises(OutputWriteError, match="Could not write CSV output"):
            write(failing, target)

        assert read_rows(target) == [["id", "amount"], ["A", "1"]]
        assert files(tmp_path) == ["result.csv"]
