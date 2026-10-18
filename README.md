# Instrument Data Pipeline

A YAML-driven PySpark pipeline that turns raw instrument trades into daily metrics (notional, average price, volume, VWAP). Transformations, data quality rules and output are configuration; the engine is generic.

## Pipeline

```
CSV ─► Reader ─► Normalizer ─► Input data quality ─► Transformations ─► Output validation ─► CSV writer
        (src/core/reader.py)   (rules, duplicates,      (YAML, in order)    (YAML rules)
                                FX conversion to USD)
```

Code: `src/core/` (config, engine, reader), `src/quality/` (normalizer, validator, currency), `src/transforms/` (filter, expression, aggregate + registries), `src/output/writer.py`. `PipelineEngine.run()` returns the valid records as a DataFrame; `OutputWriter` writes them.

## Run

```bash
# Locally (Python 3.10+, Java 11/17/21)
pip install -r requirements.txt
pytest tests/
python -m src.main                 # or: python -m src.main path/to/other.yaml

# Docker
docker build -t instrument-pipeline .
docker run --rm --entrypoint pytest instrument-pipeline tests/
docker run --rm -v "$(pwd)/output:/app/output" instrument-pipeline
```

The container runs as uid/gid 1000, so a mounted `./output` must be writable by it (otherwise `chmod a+w output`).

## Output and logging

- **Result:** `output/processed_instruments.csv`: one file with a header, valid rows only. Nothing else is written. A repeated run replaces it (never appends); a failed run leaves the previous file untouched.
- **Logging:** aggregated counts only, no individual records, via the standard `logging` module:

```
Input records: total=26
Invalid records: total=4 by_reason=[NOT_POSITIVE(volume)=3, NULL_VALUE(price)=1]
Duplicate records removed: total=1 by_reason=[DUPLICATE_RECORD=1]
Currency conversion failures: total=0
Filtered records: total=2 by_reason=[FILTERED_BY_FILTER_VALID_RECORDS=2]
Processed records: total=9
Pipeline status: SUCCESS elapsed=4.5s
```

## YAML operations

- **Filters:** `gt gte lt lte eq neq in not_in between is_null is_not_null contains starts_with ends_with`, nested with `and` / `or` / `not`.
- **Expressions:** `add subtract multiply divide abs round coalesce upper lower trim concat current_timestamp current_date` (`divide` returns null for a zero/null denominator).
- **Aggregations:** `sum avg min max count count_distinct first last` (`alias` required).
- **Validation checks:** `not_null positive non_negative gt gte lt lte eq neq in not_in between`.

Invalid configuration fails early with a descriptive `ConfigurationError`. To extend, register an operator, function or check in `src/transforms/` or `src/quality/validator.py` and add tests.

## Assumptions and limitations

- **FX rates are illustrative test data** (only `2024-01-15 EUR→USD = 1.10`). Prices are converted to USD before notional and aggregation. Unknown currencies and missing rates for a date are excluded and counted.
- **Exact duplicate rows are removed** after normalization; a real system should deduplicate on a transaction ID.
- **Bad records never stop the run:** invalid, duplicate, filtered and unconvertible records are excluded from the CSV and counted by reason. They are not persisted, so they cannot be reprocessed. Fatal problems (missing source, header mismatch, invalid FX table, unwritable output) fail the run.
- CSV sources and local Spark mode only. The result is written by a single task, so it must be small.
