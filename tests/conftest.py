"""
Pytest fixtures for pipeline tests.
"""
import os
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from src.main import create_spark_session

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def repo_root_cwd():
    """Run every test from the repository root, where relative config paths resolve."""
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    yield REPO_ROOT
    os.chdir(previous)


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing (same settings as the application)."""
    session = create_spark_session()
    session.conf.set("spark.sql.shuffle.partitions", "2")
    yield session
    session.stop()
