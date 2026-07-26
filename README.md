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

## Loading your own data

Real CSVs are messy. `vizlib.load` reads one into a clean, plot-ready
DataFrame — recognising many missing-value tokens, stripping currency
symbols / thousands separators / `%` from numeric-looking columns, parsing
date columns, and falling back from UTF-8 to Latin-1 encoding:

```python
import vizlib

df = vizlib.load("mydata.csv")     # NA tokens, "$1,234.50", dates all handled
vizlib.plots.bar(df, "department")
```

`load` stays pandas-only, so `import vizlib` remains light. Handy options:

- `numeric="auto"` (default) coerces mostly-numeric object columns and leaves
  ID/text columns alone; pass `False` to skip, or a list of column names.
- `parse_dates="auto"` (default) parses date-like columns; pass `False` or a
  list.
- `na_values=[...]` adds to the built-in tokens (`""`, `NA`, `N/A`, `null`,
  `none`, `unknown`, `?`, case-insensitive).
- `sample=1000, random_state=0` returns a reproducible subset of large files.

Even without `load`, the plots coerce numeric-looking string columns at plot
time, so `hist`/`scatter`/`line` on a `"$1,234"` column just work.

## Demo datasets

Five small, **synthetic** healthcare datasets live in
[`datasets/`](datasets/) (ER visits, oncology intake, biometrics, hospital
claims, wearable monitoring) with a data dictionary in
[`datasets/README.md`](datasets/README.md). They cover the full `vizlib`
surface and contain the messy features (`$`/comma numbers, several NA tokens,
date strings, ordered + high-cardinality categoricals, skew, outliers, gaps)
that `load` and the plots handle. Render one figure per function with:

```bash
python examples/demo.py    # writes PNGs to examples/output/
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

# Value-counts bar chart (graphical twin of value_counts_bar): top-N + "Other",
# sorted, zero baseline, bars labelled directly. Pass your finding as the title
# and use highlight= to draw the eye with the action color.
plots.bar(
    df, "city",
    title="Most orders ship from the SF hub",
    subtitle="FY24 orders, n = 4,812",
    highlight="SF",
    source="Source: fulfillment log",
)

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

The defaults follow the [Evergreen & Emery Data Visualization
Checklist](https://stephanieevergreen.com/data-visualization-checklist/):
left-justified descriptive titles, an optional muted subtitle and `source`
caption, a readable font hierarchy, dark high-contrast text, muted
gridlines, direct data labels on bars (with the redundant value axis
hidden), frequency/median-ordered categories, and honest zero-based
magnitude axes. The mechanical rules are automatic; the interpretive ones
are hooks — `title`, `subtitle`, `source`, `highlight` — with neutral
defaults. **Override `title`/`subtitle` with your actual finding**; the
default titles are placeholders and vizlib never invents a takeaway.

> The legacy `pip install "vizlib[plot]"` extra still works but is no longer
> necessary — plotting is installed by default.

## API

### Core

| Function | What it does |
| --- | --- |
| `load(path, *, parse_dates="auto", numeric="auto", na_values=None, sample=None, random_state=0, **read_csv_kwargs)` | Read a CSV into a clean, plot-ready DataFrame (NA tokens, currency/`%`/thousands, dates, encoding fallback, optional sampling). |
| `summarize(df)` | Per-column overview: dtype, non-null, nulls, null %, unique count. |
| `missing_values(df, only_missing=True)` | Missing-value counts and percentages, largest first. |
| `numeric_summary(df)` | `describe()` for numeric (or numeric-coercible) columns. |
| `value_counts_bar(series, top=10, width=40)` | ASCII horizontal bar chart of value counts. |
| `histogram(series, bins=10, width=40)` | ASCII histogram of a numeric Series. |

All core functions take a pandas object (or a path, for `load`) and never
mutate their input. `load` is the only one that imports nothing heavier than
pandas, so `import vizlib` stays fast.

### Plotting — `vizlib.plots`

| Function | What it does |
| --- | --- |
| `bar(data, column=None, *, top=15, sort=True, highlight=None, value_labels=True, precision=0, ...)` | Value-counts bar chart: top-N + "Other", sorted, zero baseline, bars labelled directly. |
| `hist(series, *, bins="auto", kde=False, ...)` | Histogram of a numeric Series, optional KDE; zero-based count axis. |
| `distribution(series, *, ...)` | Histogram + KDE + rug for a quick distribution read. |
| `box(df, column=None, *, by=None, ...)` | Boxplot for spread/outliers; groups ordered by median. |
| `scatter(df, x, y, *, hue=None, reg=False, sample=None, random_state=0, ...)` | Relationship between two numeric (or coercible) columns; frameless legend for `hue`; auto-samples large data. |
| `line(df, x, y, *, hue=None, ...)` | Line plot for ordered/time-series data; datetime axes and lines labelled directly. |
| `correlation_heatmap(df, *, method="pearson", annot=True, ...)` | Masked, annotated correlation matrix on a fixed `[-1, 1]` diverging scale. |
| `missing_bar(df, *, highlight=None, value_labels=True, precision=1, ...)` | Per-column percentage of missing values, largest first, labelled directly. |
| `missing_matrix(df, *, ...)` | Nullity matrix (dark cells mark missing values). |
| `pairplot(df, *, hue=None, columns=None, sample=None, random_state=0, ...)` | Scatter-matrix of numeric columns; returns the seaborn grid; auto-samples large data. |
| `set_theme(*, palette=..., accent=..., muted=..., text_color=..., grid_color=..., title_size=..., ...)` | Configure the shared, colorblind-safe look and font hierarchy. |

Every plotting function (except `set_theme`) accepts the keyword-only
checklist hooks `title=`, `subtitle=`, `source=` and, where it composes,
`ax=` — shown as `...` above. `bar` and `missing_bar` additionally take
`highlight=` (accent a label or list of labels), `value_labels=` (direct
labels, on by default) and `precision=` (label decimals). All new
parameters are keyword-only with defaults, so existing calls keep working.
Every plotting function takes a pandas object, never mutates it, and returns
the `Axes`/`Figure` it drew on.

## License

MIT — see [LICENSE](LICENSE).
