"""vizlib — tiny exploratory-data-analysis helpers for pandas.

Import the public functions straight from the package::

    import pandas as pd
    import vizlib

    df = pd.read_csv("data.csv")
    print(vizlib.summarize(df))
    print(vizlib.value_counts_bar(df["category"]))
"""

from .core import (
    histogram,
    load,
    missing_values,
    numeric_summary,
    summarize,
    value_counts_bar,
)

__version__ = "0.5.0"

__all__ = [
    "load",
    "summarize",
    "missing_values",
    "numeric_summary",
    "value_counts_bar",
    "histogram",
    "__version__",
]
