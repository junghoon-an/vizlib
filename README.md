# vizlib

MSDS-610-01 Visualization Library

`vizlib` is a tiny exploratory-data-analysis (EDA) toolkit for pandas
DataFrames. It gives you a quick, readable feel for a dataset — column
overviews, missing-value reports, and lightweight ASCII charts — in a
handful of plain Python functions. A standard install brings in **pandas,
matplotlib and seaborn**, so the publication-quality [plotting
functions](#plotting) work out of the box. `import vizlib` itself stays
fast: matplotlib is only loaded when you explicitly reach for
`vizlib.plots`.

## Install

Install from a local clone (editable, recommended while developing):

```bash
pip install -e .
```

Or a regular install:

```bash
pip install .
```

Both commands install the full plotting stack (pandas, matplotlib and
seaborn) — no extra step is needed to use `vizlib.plots`.

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

## Plotting

The plotting functions are backed by matplotlib and seaborn, both installed
by the standard `pip install` above. They live in their own module,
`vizlib.plots`, and are **not** re-exported from the top level — that keeps
`import vizlib` fast, since matplotlib (a slow import) loads only when you
reach for the plotting module. Import it explicitly:

```python
import pandas as pd
import matplotlib.pyplot as plt
from vizlib import plots

df = pd.read_csv("data.csv")

# Value-counts bar chart (graphical twin of value_counts_bar):
# top-N categories, the rest folded into "Other", sorted, zero baseline.
plots.bar(df, "city")

# Distribution of a numeric column, with a KDE overlay.
plots.hist(df["age"], kde=True)

# Correlation heatmap: masked upper triangle, annotated, scale fixed to [-1, 1].
plots.correlation_heatmap(df)

plt.show()  # you call show(); vizlib only ever returns the Axes/Figure
```

Every plot function **returns** the matplotlib `Axes` it drew on (or a
`Figure`/seaborn grid for multi-panel plots) and never calls `plt.show()`,
so plots compose in notebooks and subplot grids — pass `ax=` to draw into
an existing axis. Call `plots.set_theme(...)` once to customize the shared
look; the defaults are already colorblind-safe and de-cluttered.

> The legacy `pip install "vizlib[plot]"` extra still works but is no longer
> necessary — plotting is installed by default.

## API

### Core

| Function | What it does |
| --- | --- |
| `summarize(df)` | Per-column overview: dtype, non-null, nulls, null %, unique count. |
| `missing_values(df, only_missing=True)` | Missing-value counts and percentages, largest first. |
| `numeric_summary(df)` | `describe()` for numeric columns, one row per column. |
| `value_counts_bar(series, top=10, width=40)` | ASCII horizontal bar chart of value counts. |
| `histogram(series, bins=10, width=40)` | ASCII histogram of a numeric Series. |

All core functions take a pandas object and never mutate their input.

### Plotting — `vizlib.plots`

| Function | What it does |
| --- | --- |
| `bar(data, column=None, *, top=15, sort=True, ax=None)` | Value-counts bar chart with top-N + "Other", sorted, zero baseline. |
| `hist(series, *, bins="auto", kde=False, ax=None)` | Histogram of a numeric Series, optional KDE. |
| `distribution(series, *, ax=None)` | Histogram + KDE + rug for a quick distribution read. |
| `box(df, column=None, *, by=None, ax=None)` | Boxplot for spread/outliers, optionally grouped by a category. |
| `scatter(df, x, y, *, hue=None, reg=False, ax=None)` | Relationship between two numeric columns, optional regression line. |
| `line(df, x, y, *, hue=None, ax=None)` | Line plot for ordered/time-series data. |
| `correlation_heatmap(df, *, method="pearson", annot=True, ax=None)` | Masked, annotated correlation matrix on a fixed `[-1, 1]` diverging scale. |
| `missing_bar(df, *, ax=None)` | Per-column percentage of missing values, largest first. |
| `missing_matrix(df, *, ax=None)` | Nullity matrix (dark cells mark missing values). |
| `pairplot(df, *, hue=None, columns=None)` | Scatter-matrix of numeric columns; returns the seaborn grid. |
| `set_theme(*, palette="colorblind", context="notebook", ...)` | Configure the shared, colorblind-safe look. |

Every plotting function takes a pandas object, never mutates it, and
returns the `Axes`/`Figure` it drew on.

## License

MIT — see [LICENSE](LICENSE).
