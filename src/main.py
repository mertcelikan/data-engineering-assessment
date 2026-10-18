"""
Entry point for running the transformation pipeline.

Usage: ``python -m src.main [path/to/pipeline.yaml]``
"""
import logging
import sys
import time

from pyspark.sql import SparkSession

from src.core import PipelineConfig, PipelineEngine
from src.core.exceptions import PipelineError
from src.output.writer import OutputWriter

DEFAULT_CONFIG_PATH = "config/pipeline.yaml"

logger = logging.getLogger("pipeline")


def create_spark_session() -> SparkSession:
    """Create a local Spark session for development."""
    session = (
        SparkSession.builder
        .appName("instrument_pipeline")
        .master("local[*]")
        # Same behaviour locally and in Docker.
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    return session


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = sys.argv[1:] if argv is None else argv
    config_path = args[0] if args else DEFAULT_CONFIG_PATH

    started = time.monotonic()
    succeeded = False
    spark = None
    try:
        # Validate the configuration before starting Spark.
        config = PipelineConfig.from_yaml(config_path)
        spark = create_spark_session()

        result_df = PipelineEngine(spark, config).run()
        OutputWriter(config.output).write(result_df)
        succeeded = True
        return 0
    except PipelineError as exc:
        logger.error("Pipeline error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected pipeline error")
        return 1
    finally:
        logger.log(
            logging.INFO if succeeded else logging.ERROR,
            "Pipeline status: %s elapsed=%.1fs",
            "SUCCESS" if succeeded else "FAILED",
            time.monotonic() - started,
        )
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
