"""Small keyword-only config objects that collapse cohesive plot knobs.

Grouping the captions and the value-label formatting into one object each keeps
the chart functions — and the private helpers they thread through — narrow.
Both are plain dataclasses with sensible defaults, so the simplest call stays
simple and the fields read the same as the old keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Captions:
    """The left-justified title, muted subtitle and source caption of a chart."""

    title: str | None = None
    subtitle: str | None = None
    source: str | None = None


@dataclass
class ValueLabels:
    """How on-data bar labels are formatted (see ``bar``/``missing_bar``)."""

    show: bool = True
    precision: int = 0
    fmt: str | None = None
    padding: int = 5

    def format(self, values, *, percent: bool = False) -> list[str]:
        """Render ``values`` to label strings with this config."""
        used = self.fmt or (f"%.{self.precision}f%%" if percent
                            else f"%.{self.precision}f")
        return [used % v for v in values]
