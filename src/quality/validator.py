"""
Rule-based validation and duplicate detection.

``VALIDATION_CHECKS`` is the registry of checks usable in ``data_quality.input_rules``
and ``validation.rules``. To add a check: register it here and add tests.

``RuleValidator`` is stage-agnostic (used for both input and output validation);
``DataQualityValidator`` is the input-stage validator (rules + exact duplicates).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.quality.rejections import (
    DUPLICATE_RECORD,
    ERRORS_COLUMN,
    SplitResult,
    Stage,
    collect_errors,
    error,
    rejected_frame,
    split_on_errors,
)
from src.transforms.base import require_columns
from src.transforms.operators import CONDITION_OPERATORS, Predicate, ValueKind

if TYPE_CHECKING:
    from src.core.config import DataQualityConfig, RuleSpec


@dataclass(frozen=True)
class ValidationCheck:
    """A validation check: the predicate a value must satisfy and its rejection code."""

    predicate: Predicate
    reason_code: str


def _from_operator(name: str) -> ValidationCheck:
    return ValidationCheck(CONDITION_OPERATORS[name], f"CHECK_{name.upper()}_FAILED")


VALIDATION_CHECKS: dict[str, ValidationCheck] = {
    "not_null": ValidationCheck(
        Predicate(ValueKind.NONE, lambda column, _: column.isNotNull()), "NULL_VALUE"
    ),
    "positive": ValidationCheck(
        Predicate(ValueKind.NONE, lambda column, _: column > 0), "NOT_POSITIVE"
    ),
    "non_negative": ValidationCheck(
        Predicate(ValueKind.NONE, lambda column, _: column >= 0), "NEGATIVE_VALUE"
    ),
    **{
        name: _from_operator(name)
        for name in ("gt", "gte", "lt", "lte", "eq", "neq", "in", "not_in", "between")
    },
}


class RuleValidator:
    """
    Validates a DataFrame against a list of rules and quarantines violating rows.

    A null value never satisfies a rule. Each row is rejected at most once, under the
    first failing rule (code: the check's reason code, detail: the column name).
    """

    def __init__(self, rules: Sequence["RuleSpec"], stage: Stage):
        self._rules = tuple(rules)
        self._stage = stage

    def validate(self, df: DataFrame) -> SplitResult:
        require_columns(
            df, [rule.column for rule in self._rules], f"{self._stage.value} rules"
        )
        failures = []
        for rule in self._rules:
            check = VALIDATION_CHECKS[rule.check]
            passed = F.coalesce(
                check.predicate.build(F.col(rule.column), rule.value), F.lit(False)
            )
            failures.append(F.when(~passed, error(check.reason_code, rule.column)))
        checked = df.withColumn(ERRORS_COLUMN, collect_errors(*failures))
        return split_on_errors(checked, self._stage)


class DataQualityValidator:
    """
    Input data quality stage: rule validation, then exact-duplicate removal.

    Duplicates are only searched among records that passed the rules, so every source
    record is quarantined at most once. Two records are duplicates only if *all* columns
    are equal after normalisation (not merely the same instrument and date: several
    legitimate trades can share those). Duplicates are identical, so which occurrence is
    kept does not matter and no record id is needed. A real transaction system should
    deduplicate on a unique transaction id instead.
    """

    def __init__(self, config: "DataQualityConfig"):
        self._config = config

    def validate(self, df: DataFrame) -> SplitResult:
        rules = RuleValidator(self._config.input_rules, Stage.INPUT_DATA_QUALITY).validate(df)
        deduplicated = self._remove_duplicates(rules.valid)
        return SplitResult(
            valid=deduplicated.valid,
            rejected=rules.rejected.unionByName(deduplicated.rejected),
        )

    @staticmethod
    def _remove_duplicates(df: DataFrame) -> SplitResult:
        # The constant ordering only satisfies row_number(); ties are identical records.
        occurrence = F.row_number().over(Window.partitionBy(*df.columns).orderBy(F.lit(1)))
        ranked = df.withColumn("_occurrence", occurrence)
        return SplitResult(
            valid=ranked.filter(F.col("_occurrence") == 1).drop("_occurrence"),
            rejected=rejected_frame(
                ranked.filter(F.col("_occurrence") > 1), Stage.INPUT_DATA_QUALITY, DUPLICATE_RECORD
            ),
        )
