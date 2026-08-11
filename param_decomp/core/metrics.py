"""Typed, transport-independent metric records."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BarChart:
    rows: tuple[tuple[str, float], ...]
    x_label: str
    y_label: str
    title: str


@dataclass(frozen=True)
class PNGImage:
    encoded: bytes


type MetricValue = float | BarChart | PNGImage
type LogRecord = Mapping[str, MetricValue]
