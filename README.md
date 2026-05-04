# Toronto Housing Price Analysis

A Python data analysis project for cleaning and exploring Greater Toronto Area housing listing data.

The project takes raw CSV files from the downloaded GTA housing dataset, prepares one combined dataset, creates visualizations, and trains simple baseline models to estimate listing price from available listing features.

## Tools Used

- Python
- Pandas
- scikit-learn
- Matplotlib

## Project Workflow

1. Combine the raw GTA housing CSV files into one dataset.
2. Parse useful fields from raw listing text, including price, bedrooms, bathrooms, property type, city, and snapshot date.
3. Clean duplicates, missing values, inconsistent categories, and extreme price outliers.
4. Create exploratory charts.
5. Train baseline regression models and compare their performance.
6. Save cleaned data, charts, model metrics, and a summary report.

## Dataset

The raw data is expected in this folder:

```text
greater-toronto-area-housing-data-master/
  greater-toronto-area-housing-data-master/
    data/
```

The downloaded CSV files contain:

| Column | Description |
| --- | --- |
| `address` | Listing address |
| `price` | Listing price as text, such as `C$539,000` |
| `details` | Listing details, such as `2 bds2 ba- Condo for sale` |

The preparation script creates additional columns from those fields:

| Column | Description |
| --- | --- |
| `bedrooms` | Parsed bedroom count |
| `bathrooms` | Parsed bathroom count |
| `square_feet` | Parsed size when available |
| `property_type` | Parsed listing type, such as condo, townhouse, or house |
| `city` | City from the source folder |
| `snapshot_date` | Date from the source filename |

## Quick Start

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Prepare the real GTA data:

```powershell
py -m toronto_housing_analysis.prepare_gta_data
```

Run the analysis pipeline:

```powershell
py -m toronto_housing_analysis.pipeline
```

## Outputs

After running the project, generated files are saved to:

```text
data/raw/toronto_housing.csv
data/processed/cleaned_toronto_housing.csv
reports/model_metrics.csv
reports/feature_importance.csv
reports/analysis_summary.md
reports/figures/
```

## Notes

The source dataset has limited fields. Most rows include address, price, bedrooms, bathrooms, and property type. Square footage appears only in some listings, so size-based analysis may be limited.
