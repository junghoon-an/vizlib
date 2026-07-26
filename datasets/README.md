# vizlib demo datasets

Five small, **entirely synthetic** healthcare datasets used to demo `vizlib`.
They contain no real patients, facilities, or providers — every value is
randomly generated and illustrative only. They deliberately include the messy
features (currency-formatted numbers, several missing-value tokens, date
strings, ordered and high-cardinality categoricals, skew, outliers, gaps) that
`vizlib.load` and the plotting layer are built to handle, so they double as
test fixtures.

Load any file with the cleaning built into `vizlib.load`:

```python
import vizlib
df = vizlib.load("datasets/er_daily_visits.csv")   # NA tokens, $ / commas, dates handled
```

Missing values are encoded with a mix of tokens across the files (`""`, `NA`,
`N/A`, `null`, `unknown`, `?`); `vizlib.load` maps all of them to `NaN`.

---

## `er_daily_visits.csv` — ER operations (time series + correlations)

120 rows (30 days × 4 departments).

| Column | Type | Notes |
| --- | --- | --- |
| `date` | date string | `YYYY-MM-DD`, parsed to datetime by `load`. |
| `department` | category | ER, Cardiology, Pediatrics, Orthopedics. |
| `admissions` | int | Daily admissions. |
| `billed_amount` | currency string | e.g. `"$18,240.50"` → numeric by `load`. |
| `avg_wait_min` | float | Some values are the token `N/A`. |
| `staff_on_duty` | int | |
| `weekend` | Yes/No | |

**Demo:** `line` (admissions over `date`, `hue=department`), `scatter`
(`avg_wait_min` vs `admissions`), `correlation_heatmap` (admissions/wait/staff),
`bar` (department), `box` (admissions by department). *So what?* Wait times rise
with admissions, and the ER dominates volume — especially on weekends.

## `patient_intake.csv` — oncology intake (missingness, high cardinality, ordered categorical)

150 rows.

| Column | Type | Notes |
| --- | --- | --- |
| `patient_id` | id | `PT-0001` … |
| `age` | int | ~12% blank (`""`). |
| `city` | category | ~30 distinct → use `bar(..., top=N)`. |
| `stage` | ordered category | Cancer stage `I < II < III < IV`. |
| `treatment_cost_usd` | thousands string | e.g. `"41,203"`; ~20% missing. |
| `satisfaction` | int 1–5 | Some values are the token `null`. |
| `diagnosis_date` | date string | |

**Demo:** `summarize`, `missing_values` / `missing_bar` / `missing_matrix`,
`bar` (city, top-N), `box` (treatment cost by `stage`, ordered groups), `hist`
(age). *So what?* Cost climbs with stage; missingness clusters in cost and age.

## `patient_vitals.csv` — biometrics (multivariate numeric relationships)

150 rows.

| Column | Type | Notes |
| --- | --- | --- |
| `patient_id` | id | |
| `risk_group` | category | Healthy / Prediabetic / Diabetic. |
| `clinic` | category | North / South / East / West. |
| `weight_kg`, `bmi`, `systolic_bp`, `diastolic_bp`, `cholesterol`, `glucose`, `resting_hr` | numeric | Correlated; shifted upward by risk group. |
| `sex` | category | ~7% missing (token `?`). |

**Demo:** `pairplot` (`hue=risk_group`), `correlation_heatmap`, `scatter`
(`bmi` vs `glucose`, `hue`, `reg=True`), `distribution` (glucose), `box`
(glucose by `risk_group`). *So what?* Metabolic markers move together and
separate the risk groups.

## `hospital_claims.csv` — billing (skewed distributions + mixed types)

180 rows.

| Column | Type | Notes |
| --- | --- | --- |
| `claim_id` | id | |
| `diagnosis_category` | category | |
| `admission_year` | ordered int | 2019–2023. |
| `length_of_stay_days` | int | Right-skewed; some values are the token `unknown`. |
| `total_charges_usd` | currency string | Right-skewed; e.g. `"12,043.90"`. |
| `reimbursement_usd` | currency string | Correlated with charges. |
| `readmitted` | Yes/No | |
| `insurance_provider` | category | High cardinality (16 providers). |

**Demo:** `hist` (skewed `total_charges_usd`), `scatter` (charges vs
reimbursement), `bar` (diagnosis category), `line` (mean charges by
`admission_year`), `missing_bar`. *So what?* Charges are heavy-tailed and
reimbursement tracks them imperfectly.

## `patient_monitoring.csv` — wearable/bedside (grouped time series + outliers + gaps)

200 rows (50 timestamps × 4 patients).

| Column | Type | Notes |
| --- | --- | --- |
| `timestamp` | datetime string | `YYYY-MM-DD HH:MM:SS`. |
| `patient_id` | group | A / B / C / D. |
| `heart_rate_bpm` | float | Occasional outliers; ~6% gaps (token `NA`). |
| `spo2_pct` | float | ~5% blank (`""`). |
| `battery_pct` | float | Declines over time. |
| `activity` | category | Resting / Active / Sleeping. |

**Demo:** `line` (heart rate over `timestamp`, `hue=patient_id`), `box` (heart
rate by patient), `distribution` (spo2), `missing_matrix`, `scatter` (spo2 vs
heart rate). *So what?* Heart-rate baselines differ by patient, with sporadic
outliers and monitoring gaps.

---

*Generated once as static files — there is no generator script in the repo.*
