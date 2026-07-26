# vizlib

MSDS-610-01 Visualization Library

`vizlib` is a tiny exploratory-data-analysis (EDA) toolkit for pandas
DataFrames. It gives you a quick, readable feel for a dataset — column
overviews, missing-value reports, and no-dependency ASCII charts — in a
handful of plain Python functions. One dependency: **pandas**.

## Install

Install from a local clone (editable, recommended while developing):

```bash
pip install -e .
```

Or a regular install:

```bash
pip install .
```

## Quick start

```python
import pandas as pd
import vizlib

df = pd.DataFrame(
    {
        "city": ["SF", "SF", "LA", "NYC", None],
        "age": [21, 34, 29, 41, 25],
    }
)

# One-row-per-column overview: dtype, non_null, nulls, null_pct, unique
print(vizlib.summarize(df))

# Just the columns that have missing values, worst first
print(vizlib.missing_values(df))

# describe() for numeric columns only
print(vizlib.numeric_summary(df))

# ASCII bar chart of a categorical column
print(vizlib.value_counts_bar(df["city"]))

# ASCII histogram of a numeric column
print(vizlib.histogram(df["age"], bins=4))
```

## API

| Function | What it does |
| --- | --- |
| `summarize(df)` | Per-column overview: dtype, non-null, nulls, null %, unique count. |
| `missing_values(df, only_missing=True)` | Missing-value counts and percentages, largest first. |
| `numeric_summary(df)` | `describe()` for numeric columns, one row per column. |
| `value_counts_bar(series, top=10, width=40)` | ASCII horizontal bar chart of value counts. |
| `histogram(series, bins=10, width=40)` | ASCII histogram of a numeric Series. |

All functions take a pandas object and never mutate their input.

## License

MIT — see [LICENSE](LICENSE).
