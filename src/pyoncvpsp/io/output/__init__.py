"""ONCVPSP output I/O package."""

from ._data import ERRORS, WARNINGS
from ._messages import (
    Attribution,
    Level,
    OncvpspFailure,
    OncvpspMessage,
    Scope,
    Stage,
)
from ._attribution import ATTRIBUTION, analyze_failure, attribute_messages
from ._sections import line_contexts
from ._text import OncvpspOutputError, OncvpspTextParser

__all__: list[str] = [
    "ATTRIBUTION",
    "Attribution",
    "ERRORS",
    "Level",
    "OncvpspFailure",
    "OncvpspMessage",
    "OncvpspTextParser",
    "OncvpspOutputError",
    "Scope",
    "Stage",
    "WARNINGS",
    "analyze_failure",
    "attribute_messages",
    "line_contexts",
]
