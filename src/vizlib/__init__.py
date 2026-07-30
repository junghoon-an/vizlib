"""vizlib — tiny exploratory-data-analysis helpers for pandas.

Import the public functions straight from the package::

    import pandas as pd
    import vizlib

    df = pd.read_csv("data.csv")
    print(vizlib.summarize(df))
    print(vizlib.value_counts_bar(df["category"]))
"""

from .ascii import histogram, value_counts_bar
from .eda import missing_values, numeric_summary, summarize
from .loading import load

__version__ = "0.7.1"

__all__ = [
    "load",
    "summarize",
    "missing_values",
    "numeric_summary",
    "value_counts_bar",
    "histogram",
    "__version__",
]
