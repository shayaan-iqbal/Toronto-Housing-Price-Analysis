# Toronto Housing Price Analysis

Portfolio-ready Python project for cleaning, exploring, and modeling Toronto housing prices with Pandas, scikit-learn, and Matplotlib.

## What this project does

- Loads a Toronto housing CSV from `data/raw/toronto_housing.csv`
- Cleans missing values, duplicate rows, inconsistent categories, and obvious price/size outliers
- Builds exploratory plots for price distribution, neighborhood medians, property type mix, and price vs. size
- Trains baseline regression models and reports MAE, RMSE, and R2
- Saves cleaned data, figures, model metrics, and a short analysis summary

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m toronto_housing_analysis.prepare_gta_data
python -m toronto_housing_analysis.pipeline
```

If `python` is not recognized on Windows, try the same commands with `py`:

```powershell
py -m toronto_housing_analysis.prepare_gta_data
py -m toronto_housing_analysis.pipeline
```

If the Microsoft Store Python gives WindowsApps or virtual-environment errors, install Python from python.org, then reopen PowerShell and rerun the quick start.

The real-data preparation script reads the downloaded `greater-toronto-area-housing-data-master` folder and creates `data/raw/toronto_housing.csv`.

If you do not have the real data folder yet, the sample generator creates `data/raw/toronto_housing_sample.csv` with 6,000 synthetic rows:

```powershell
python -m toronto_housing_analysis.generate_sample_data
python -m toronto_housing_analysis.pipeline
```

Important: do not describe the sample file as real Toronto market data on a resume. Use it to build and test the project, then replace it with a real CSV named `data/raw/toronto_housing.csv`.

## Expected CSV columns

The pipeline accepts common column-name variants, but these are the clean target names:

| Column | Meaning |
| --- | --- |
| `price` | Sale or listing price in CAD |
| `bedrooms` | Bedroom count |
| `bathrooms` | Bathroom count |
| `square_feet` | Interior size |
| `property_type` | Condo, detached, semi-detached, townhouse, etc. |
| `neighborhood` | Toronto neighborhood or district |
| `parking` | Parking space count |
| `days_on_market` | Days listed before sale/removal |

Only `price` is strictly required, but the analysis is stronger when all columns are present.

The downloaded GTA dataset only includes `address`, `price`, and `details`, so `prepare_gta_data.py` parses bedrooms, bathrooms, occasional square-foot values, property type, city, and snapshot date from those fields.

## Outputs

After running the pipeline, check:

- `data/processed/cleaned_toronto_housing.csv`
- `reports/figures/*.png`
- `reports/model_metrics.csv`
- `reports/feature_importance.csv`
- `reports/analysis_summary.md`

## Resume wording

Once you replace the sample file with a real 5,000+ row dataset, your bullet can honestly be:

> Cleaned and analyzed 5,000+ Toronto housing records with reproducible preprocessing for missing values, outliers, and category standardization.
>
> Performed exploratory analysis and trained baseline regression models to evaluate key drivers of price variation using MAE, RMSE, R2, and clear Matplotlib visualizations.
