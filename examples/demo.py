"""Render one representative figure per vizlib function from the demo data.

Loads each synthetic healthcare CSV in ``datasets/`` with ``vizlib.load`` and
saves a PNG per plot to ``examples/output/``. Run from the repo root::

    python examples/demo.py

All data is synthetic (see ``datasets/README.md``).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # non-interactive: save files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import vizlib  # noqa: E402
from vizlib import plots  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "datasets")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)


def save(ax_or_grid, name: str) -> None:
    """Save an Axes or a seaborn grid to OUT/<name>.png, then close it."""
    fig = getattr(ax_or_grid, "figure", None) or ax_or_grid.fig
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # 1) ER daily visits — line, scatter, correlation, bar, box
    er = vizlib.load(os.path.join(DATA, "er_daily_visits.csv"))
    save(plots.line(er, "date", "admissions", hue="department",
                    title="ER admissions climb on weekends"), "er_line")
    save(plots.scatter(er, "admissions", "avg_wait_min",
                       title="Wait time rises with admissions"), "er_scatter")
    save(plots.correlation_heatmap(er[["admissions", "avg_wait_min", "staff_on_duty"]]),
         "er_corr")
    save(plots.bar(er, "department", title="ER is the busiest department"), "er_bar")
    save(plots.box(er, "admissions", by="department"), "er_box")

    # 2) Patient intake — summarize/missingness, bar (top-N), box (ordered), hist
    intake = vizlib.load(os.path.join(DATA, "patient_intake.csv"))
    print(vizlib.summarize(intake))
    print(vizlib.missing_values(intake))
    save(plots.missing_bar(intake), "intake_missing_bar")
    save(plots.missing_matrix(intake), "intake_missing_matrix")
    save(plots.bar(intake, "city", top=8, title="Intake concentrates in a few cities"),
         "intake_bar")
    intake["stage"] = pd.Categorical(intake["stage"], ["I", "II", "III", "IV"], ordered=True)
    save(plots.box(intake, "treatment_cost_usd", by="stage",
                   title="Treatment cost rises with stage"), "intake_box")
    save(plots.hist(intake["age"], kde=True), "intake_hist")

    # 3) Patient vitals — pairplot, correlation, scatter (reg), distribution, box
    vitals = vizlib.load(os.path.join(DATA, "patient_vitals.csv"))
    save(plots.pairplot(vitals, hue="risk_group",
                        columns=["bmi", "glucose", "cholesterol"]), "vitals_pairplot")
    save(plots.correlation_heatmap(vitals), "vitals_corr")
    save(plots.scatter(vitals, "bmi", "glucose", hue="risk_group", reg=True,
                       title="Glucose rises with BMI"), "vitals_scatter")
    save(plots.distribution(vitals["glucose"]), "vitals_distribution")
    save(plots.box(vitals, "glucose", by="risk_group"), "vitals_box")

    # 4) Hospital claims — hist (skewed), scatter, bar, line, missing_bar
    claims = vizlib.load(os.path.join(DATA, "hospital_claims.csv"))
    save(plots.hist(claims["total_charges_usd"], title="Charges are right-skewed"),
         "claims_hist")
    save(plots.scatter(claims, "total_charges_usd", "reimbursement_usd"), "claims_scatter")
    save(plots.bar(claims, "diagnosis_category"), "claims_bar")
    save(plots.line(claims.groupby("admission_year", as_index=False)["total_charges_usd"].mean(),
                    "admission_year", "total_charges_usd",
                    title="Mean charges by admission year"), "claims_line")
    save(plots.missing_bar(claims), "claims_missing_bar")

    # 5) Patient monitoring — line (grouped time series), box, distribution, matrix, scatter
    mon = vizlib.load(os.path.join(DATA, "patient_monitoring.csv"))
    save(plots.line(mon, "timestamp", "heart_rate_bpm", hue="patient_id",
                    title="Heart rate over time by patient"), "mon_line")
    save(plots.box(mon, "heart_rate_bpm", by="patient_id"), "mon_box")
    save(plots.distribution(mon["spo2_pct"]), "mon_distribution")
    save(plots.missing_matrix(mon), "mon_missing_matrix")
    save(plots.scatter(mon, "heart_rate_bpm", "spo2_pct"), "mon_scatter")

    print(f"\nSaved figures to {os.path.relpath(OUT)}/")


if __name__ == "__main__":
    main()
