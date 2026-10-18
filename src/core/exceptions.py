"""
Pipeline exception hierarchy.

Fatal problems (configuration, source, reference data, schema, output I/O) raise a
subclass of ``PipelineError``. Record-level data quality problems never raise: those
records are excluded from the result and counted in the log instead.
"""


class PipelineError(Exception):
    """Base class for all fatal pipeline errors."""


class ConfigurationError(PipelineError):
    """The YAML configuration is malformed or uses an unsupported operation."""


class SourceReadError(PipelineError):
    """The source data could not be found or read."""


class ReferenceDataError(PipelineError):
    """Reference data (e.g. FX rates) is missing or internally inconsistent."""


class SchemaError(PipelineError):
    """A column required by the configuration is missing from the data."""


class OutputWriteError(PipelineError):
    """Pipeline outputs could not be written."""
