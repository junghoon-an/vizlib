# vizlib

MSDS-610-01 Visualization Library

`vizlib` is a tiny exploratory-data-analysis toolkit for pandas DataFrames:
quick column overviews, missing-value reports, ASCII charts, and
publication-quality matplotlib/seaborn [plots](#plotting). `import vizlib`
stays fast — matplotlib loads only when you reach for `vizlib.plots`.

## Install

```bash
pip install -e .     # editable, recommended while developing
# or: pip install .
```

A standard install pulls in **pandas, matplotlib and seaborn**, so
`vizlib.plots` works out of the box. After pulling new changes, reinstall so
updates (e.g. new style presets) are picked up.

## Quick start

```python
import pandas as pd
import vizlib

df = pd.DataFrame({"city": ["SF", "SF", "LA", "NYC", None], "age": [21, 34, 29, 41, 25]})

print(vizlib.summarize(df))          # dtype, non_null, nulls, null_pct, unique
print(vizlib.missing_values(df))     # columns with gaps, worst first
print(vizlib.numeric_summary(df))    # describe() for numeric columns
print(vizlib.value_counts_bar(df["city"]))   # ASCII bar chart
print(vizlib.histogram(df["age"], bins=4))   # ASCII histogram
```

## Loading your own data

`vizlib.load` reads a messy CSV into a clean, plot-ready DataFrame —
recognising many NA tokens, stripping `$`/commas/`%` from numeric-looking
columns, parsing dates, and falling back from UTF-8 to Latin-1:

```python
df = vizlib.load("mydata.csv")
vizlib.plots.bar(df, "department")
```

Options: `numeric="auto"` and `parse_dates="auto"` (pass `False` or a column
list to control), `na_values=[...]` (adds to the built-in tokens),
`sample=1000, random_state=0` (reproducible subset of large files). `load` is
pandas-only, so `import vizlib` stays light; the plots also coerce
numeric-looking strings (`"$1,234"`) at plot time.

## Demo datasets

Five small **synthetic** healthcare CSVs live in [`datasets/`](datasets/) (ER
visits, oncology intake, biometrics, hospital claims, wearable monitoring).
They exercise the whole surface and the messy features `load` and the plots
handle (`$`/comma numbers, NA tokens, date strings, ordered and
high-cardinality categoricals, skew, outliers, gaps). See
[`datasets/README.md`](datasets/README.md) for the per-file data dictionary.

## Plotting

Plots live in `vizlib.plots` — **not** re-exported from the top level, so
`import vizlib` stays fast (matplotlib loads only on `from vizlib import
plots`). Every function **returns** the `Axes` it drew on (or a
`Figure`/seaborn grid) and never calls `plt.show()`, so plots compose in
notebooks and subplot grids — pass `ax=` to draw into an existing axis.

```python
import matplotlib.pyplot as plt
import vizlib
from vizlib import plots

df = vizlib.load("datasets/er_daily_visits.csv")

plots.bar(df, "department",
          title="ER volume concentrates in a few departments",
          highlight="Cardiology", source="Source: intake log")
plots.hist(df["admissions"], kde=True)
plots.line(df, "date", "admissions")
plots.correlation_heatmap(df)
plt.show()   # you call show(); vizlib only ever returns the Axes/Figure
```

Defaults follow the [Evergreen & Emery Data Visualization
Checklist](https://stephanieevergreen.com/data-visualization-checklist/):
left-justified titles, optional muted `subtitle`/`source`, direct bar labels,
frequency/median-ordered categories, honest zero-based axes, and a
colorblind-safe palette. The mechanical rules are automatic; the interpretive
hooks — `title`, `subtitle`, `source`, `highlight` — have neutral
placeholders you should **override with your actual finding**.

> The legacy `pip install "vizlib[plot]"` extra still works but is no longer
> needed — plotting is installed by default.

## Styling / themes

Call `set_theme` once to switch the whole look. Three presets:

- `"default"` — the colorblind-safe, de-cluttered checklist style (above).
- `"infographic"` — a bold, vivid dashboard on white: saturated palette, big
  on-data labels, gradient area fills, swatch legends, borderless chrome.
- `"neon"` — the same bold chrome on a **dark-navy** background with a neon
  palette (pink, cyan, mint, lavender, yellow) and light text.

```python
from vizlib import plots

df = vizlib.load("datasets/er_daily_visits.csv")

plots.set_theme(style_preset="neon")          # or "infographic"
plots.bar(df, "department")
plots.line(df, "date", "admissions", area=True)

plots.set_theme(style_preset="default")       # back to the analytical look
```

The `infographic`/`neon` palettes are **not colorblind-safe** — presentation
looks, not analytical. Everything is opt-in: with no `set_theme` call, plots
look exactly as they do by default.

## Demo

Runnable from the repo root against the bundled [`datasets/`](datasets/):

```python
import matplotlib
matplotlib.rcParams["toolbar"] = "None"   # figure windows open without the nav toolbar

import matplotlib.pyplot as plt
import vizlib
from vizlib import plots

# Distributions & relationships — patient vitals
df = vizlib.load("datasets/patient_vitals.csv")
plots.distribution(df["glucose"])
subset = df[df["risk_group"].isin(["Healthy", "Diabetic"])]   # drop Prediabetic
plots.scatter(subset, "bmi", "glucose", hue="risk_group", reg=True)

# Missing-data patterns & ordered groups — patient intake
df = vizlib.load("datasets/patient_intake.csv")
plots.missing_bar(df)
plots.box(df, "treatment_cost_usd", by="stage")     # boxes run stage I -> IV

# Grouped time series — patient monitoring
df = vizlib.load("datasets/patient_monitoring.csv")
plots.line(df, "timestamp", "heart_rate_bpm", hue="patient_id")
plt.show()
```

For your own CSV, inspect first to discover the columns, then plot and save:

```python
df = vizlib.load("path/to/your.csv")
print(vizlib.summarize(df))
plots.bar(df, "some_category_column")
ax = plots.hist(df["some_numeric_column"])
ax.figure.savefig("chart.png", dpi=150, bbox_inches="tight")
```

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
mutate their input. `load` imports nothing heavier than pandas.

### Plotting — `vizlib.plots`

| Function | What it does |
| --- | --- |
| `bar(data, column=None, *, top=15, sort=True, highlight=None, value_labels=True, precision=0, label_padding=5, max_label_chars=None, ...)` | Value-counts bar: top-N + "Other", sorted, zero baseline, bars labelled past the tip. |
| `hist(series, *, bins="auto", kde=False, ...)` | Histogram of a numeric Series, optional KDE; zero-based count axis. |
| `distribution(series, *, ...)` | Histogram + KDE + rug for a quick distribution read. |
| `box(df, column=None, *, by=None, ...)` | Boxplot for spread/outliers; groups ordered by median. |
| `scatter(df, x, y, *, hue=None, reg=False, sample=None, random_state=0, annotations=None, ...)` | Two-column relationship; legend for `hue`; leader-line callouts; auto-samples large data. |
| `line(df, x, y, *, hue=None, area=False, stack=False, annotations=None, ...)` | Line/time-series; datetime axes, gradient/stacked area, callouts. |
| `correlation_heatmap(df, *, method="pearson", annot=True, ...)` | Masked, annotated correlation matrix on a fixed `[-1, 1]` scale. |
| `missing_bar(df, *, highlight=None, value_labels=True, precision=1, label_padding=5, max_label_chars=None, ...)` | Per-column % missing, largest first, labelled past each bar tip. |
| `missing_matrix(df, *, ...)` | Nullity matrix (dark cells mark missing values). |
| `pairplot(df, *, hue=None, columns=None, sample=None, random_state=0, ...)` | Scatter-matrix of numeric columns; returns the seaborn grid. |
| `set_theme(*, style_preset="default"/"infographic"/"neon", palette=..., accent=..., background=..., ...)` | Switch the whole look and tune individual knobs. |

Every plotting function (except `set_theme`) accepts the keyword-only hooks
`title=`, `subtitle=`, `source=` and, where it composes, `ax=` (the `...`
above). All new parameters are keyword-only with defaults, so existing calls
keep working; every function takes a pandas object, never mutates it, and
returns the `Axes`/`Figure`.

Horizontal-bar charts (`bar`, `missing_bar`) reserve their left margin and
right label headroom for the largest label footprint across **all** presets,
so the bars sit in the same place — and column names never overlap them —
whichever style is active. Other charts use matplotlib's constrained layout.
When you pass your own `ax=`, vizlib leaves your figure's layout untouched.

## License

MIT — see [LICENSE](LICENSE).
