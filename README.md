# Instrument Data Pipeline

A YAML-driven PySpark pipeline that turns raw financial instrument trades into daily metrics (notional, average price, volume, VWAP). Transformations, data quality rules and the output are configuration; the engine is generic.

## Architecture

```
CSV ─► 1 Reader ─► 2 Normalizer ─► 3 Input data quality ─► 4 Transformations ─► 5 Output validation ─► 6 Writer
                                       (rules, duplicates,     (YAML, in order)     (YAML rules)         one CSV file
                                        FX conversion to USD)
        invalid records are excluded from the result and counted by reason in the log
```

| Stage | Code |
|---|---|
| Configuration (typed, validated up front) | `src/core/config.py` |
| 1 Reader (all columns as strings, explicit schema) | `src/core/reader.py` |
| 2 Normalizer (trim, case, dates, decimals) | `src/quality/normalizer.py` |
| 3 Rules, duplicates, FX conversion | `src/quality/validator.py`, `src/quality/currency.py` |
| 4 Transformations + registry | `src/transforms/` |
| 5 Output validation | `src/quality/validator.py` |
| 6 CSV writer | `src/output/writer.py` |
| Orchestration and logging | `src/core/engine.py` |

`PipelineEngine.run()` returns the valid records as a lazy DataFrame and writes no files; `OutputWriter` writes it to the CSV.

## Run

```bash
# Locally (Python 3.10+, Java 11/17/21, PySpark 3.5.x)
pip install -r requirements.txt
pytest tests/
python -m src.main                 # or: python -m src.main path/to/other.yaml

# Docker
docker build -t instrument-pipeline .
docker run --rm --entrypoint pytest instrument-pipeline tests/
docker run --rm -v "$(pwd)/output:/app/output" instrument-pipeline
```

The container runs as a non-root user (uid/gid 1000), so a mounted `./output` must be writable by it (the default on Docker Desktop and for a single Linux user; otherwise `chmod a+w output`).

## Input and output

| | Path |
|---|---|
| Trades | `data/input/instruments.csv` |
| FX rates | `data/reference/fx_rates.csv` (`trade_date,from_currency,to_currency,rate`) |
| **Result** | `output/processed_instruments.csv`: one CSV with a header, valid rows only, columns in `output.columns` order |

Nothing else is written: no rejected-record files and no reports. The path is `output.path` in the YAML (must end in `.csv`).

Spark can only write a directory of `part-*` files, so the result is coalesced to one partition, written to a hidden staging directory next to the target, and the part file is atomically moved onto the target. A repeated run **replaces** the CSV (never appends), a failed run leaves the previous CSV untouched, and no staging files remain.

## Logging

Data quality statistics are logged with the standard `logging` module (anything rejected is a WARNING). They are aggregated counts only: no records are logged or collected to the driver.

```
Input records: total=26
Invalid records: total=4 by_reason=[NOT_POSITIVE(volume)=3, NULL_VALUE(price)=1]
Duplicate records removed: total=1 by_reason=[DUPLICATE_RECORD=1]
Currency conversion failures: total=0
Accepted input records: total=21
Filtered records: total=2 by_reason=[FILTERED_BY_FILTER_VALID_RECORDS=2]
Output validation failures: total=0
Processed records: total=9
Pipeline status: SUCCESS elapsed=4.8s
```

A reason is a code plus a static detail (the column name or a currency pair), never a value. A failed run logs `Pipeline status: FAILED` at ERROR level and exits with code 1.

**Spark cost of the statistics:** two small jobs. The first counts and caches the normalized data (`MEMORY_AND_DISK`, reused by every stage's valid/rejected branches); the second computes every count in one aggregation. The cache is released in a `finally` before `run()` returns. The CSV write is a separate job.

## Supported YAML operations

- **Filter operators:** `gt gte lt lte eq neq in not_in between is_null is_not_null contains starts_with ends_with`, nested with `and` / `or` / `not`.
- **Expression functions:** `add subtract multiply divide abs round coalesce upper lower trim concat current_timestamp current_date`. Inputs are `column:`, `literal:` or a nested `function:`; `round` takes `scale:`. `divide` returns null for a zero or null denominator.
- **Aggregations:** `sum avg min max count count_distinct first last`; an `alias` is required. `first`/`last` are not deterministic without an ordering; use them only for columns constant within a group (as `currency` is here).
- **Validation checks:** `not_null positive non_negative gt gte lt lte eq neq in not_in between`.

Invalid configuration fails early with a descriptive `ConfigurationError`; unknown fields are rejected.

**Extending:** register an operator in `src/transforms/operators.py`, a function in `src/transforms/functions.py` or a check in `src/quality/validator.py`, and add tests. A new transformation type needs a `Transform` subclass in `src/transforms/registry.py` and a step parser in `config.py`.

## Assumptions and limitations

- **FX rates are illustrative test data, not market data.** Only `2024-01-15 EUR→USD = 1.10` is provided. Everything is converted to USD *before* notional and aggregation. Rates must be positive and unique per date and currency pair, otherwise the run fails. A currency with no rate on any date is `UNSUPPORTED_CURRENCY`; a known currency without a rate for the trade date is `MISSING_FX_RATE`.
- **Exact duplicate rows are removed** (`DUPLICATE_RECORD`) after normalization; trades that merely share instrument and date are kept. A real system should deduplicate on a transaction ID.
- **Record-level problems never stop the run.** Invalid records (null keys or price, non-positive volume, unparseable dates or numbers), rows removed by YAML filters and output rows failing validation are excluded from the CSV and counted in the log. Fatal problems (missing source file, header mismatch, invalid FX table, unwritable output) fail the run.
- Rejected records are not persisted or logged individually, so they cannot be reprocessed from disk; only counts by reason are kept. There is no per-record source row id for the same reason (and `monotonically_increasing_id` would not be a CSV line number anyway).
- Prices are `DECIMAL(18,6)`, volumes `DECIMAL(18,4)` (more fractional digits are rounded half-up). No Python UDFs are used.
- Date formats are tried in the configured order; only list formats that cannot collide (`MM/dd/yyyy` vs `dd/MM/yyyy`).
- Only CSV sources and local Spark mode are supported. The result is written by a single task, so it must be small enough for that. `on_failure`, `on_missing_rate` and `duplicates.action` accept a single value each.
