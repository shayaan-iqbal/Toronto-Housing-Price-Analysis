from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_MAIN_PATH = DATA_DIR / "raw" / "toronto_housing.csv"
RAW_SAMPLE_PATH = DATA_DIR / "raw" / "toronto_housing_sample.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "cleaned_toronto_housing.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "price"
NUMERIC_FEATURES = ["bedrooms", "bathrooms", "square_feet", "parking", "days_on_market"]
CATEGORICAL_FEATURES = ["property_type", "neighborhood"]

COLUMN_ALIASES = {
    "price": ["price", "sold_price", "sale_price", "selling_price", "list_price", "asking_price"],
    "bedrooms": ["bedrooms", "beds", "bed", "br"],
    "bathrooms": ["bathrooms", "baths", "bath", "ba"],
    "square_feet": ["square_feet", "sqft", "sq_ft", "size_sqft", "living_area", "area_sqft"],
    "property_type": ["property_type", "home_type", "house_type", "type", "propertytype"],
    "neighborhood": ["neighborhood", "neighbourhood", "community", "district", "area"],
    "parking": ["parking", "parking_spots", "garage_spaces", "parking_spaces"],
    "days_on_market": ["days_on_market", "dom", "days_listed", "listing_days"],
}


def clean_column_name(column: str) -> str:
    column = column.strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    return column.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(column) for column in df.columns]

    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        if target in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = target
                break
    return df.rename(columns=rename_map)


def parse_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[$,]", "", regex=True)
    extracted = cleaned.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def standardize_property_type(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    if "semi" in text:
        return "Semi-Detached"
    if "detached" in text:
        return "Detached"
    if "town" in text:
        return "Townhouse"
    if "condo" in text or "apartment" in text:
        return "Condo"
    return text.title()


def standardize_category(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    return re.sub(r"\s+", " ", str(value).strip()).title()


def remove_iqr_outliers(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> tuple[pd.DataFrame, int]:
    if column not in df.columns:
        return df, 0

    values = df[column].dropna()
    if values.empty:
        return df, 0

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = max(0, q1 - multiplier * iqr)
    upper = q3 + multiplier * iqr
    before = len(df)
    filtered = df[df[column].isna() | df[column].between(lower, upper)].copy()
    return filtered, before - len(filtered)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = normalize_columns(df)
    report: dict[str, int] = {"rows_loaded": len(df)}

    if TARGET not in df.columns:
        raise ValueError(
            "Could not find a price column. Rename your target column to 'price' or one of: "
            + ", ".join(COLUMN_ALIASES["price"])
        )

    for column in [TARGET, *NUMERIC_FEATURES]:
        if column in df.columns:
            df[column] = parse_number(df[column])

    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    report["duplicate_rows_removed"] = duplicates

    missing_target = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET]).copy()
    report["rows_removed_missing_price"] = missing_target

    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].map(standardize_property_type)
    if "neighborhood" in df.columns:
        df["neighborhood"] = df["neighborhood"].map(standardize_category)

    missing_feature_values = int(df[[c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]].isna().sum().sum())
    report["missing_feature_values_imputed"] = missing_feature_values

    for column in [c for c in NUMERIC_FEATURES if c in df.columns]:
        df[column] = df[column].fillna(df[column].median())
    for column in [c for c in CATEGORICAL_FEATURES if c in df.columns]:
        df[column] = df[column].fillna("Unknown")

    df, removed_price_outliers = remove_iqr_outliers(df, TARGET, multiplier=1.75)
    df, removed_size_outliers = remove_iqr_outliers(df, "square_feet", multiplier=2.0)
    report["price_outliers_removed"] = removed_price_outliers
    report["size_outliers_removed"] = removed_size_outliers
    report["rows_after_cleaning"] = len(df)

    return df.reset_index(drop=True), report


def make_figures(df: pd.DataFrame) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df[TARGET], bins=40, color="#2f6f73", edgecolor="white")
    ax.set_title("Toronto Housing Price Distribution")
    ax.set_xlabel("Price (CAD)")
    ax.set_ylabel("Listings")
    ax.ticklabel_format(style="plain", axis="x")
    path = FIGURE_DIR / "price_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figure_paths.append(path)

    if "neighborhood" in df.columns:
        medians = df.groupby("neighborhood")[TARGET].median().sort_values().tail(12)
        fig, ax = plt.subplots(figsize=(9, 6))
        medians.plot(kind="barh", ax=ax, color="#8c5e3c")
        ax.set_title("Median Price by Neighborhood")
        ax.set_xlabel("Median price (CAD)")
        ax.ticklabel_format(style="plain", axis="x")
        path = FIGURE_DIR / "median_price_by_neighborhood.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths.append(path)

    if "square_feet" in df.columns:
        sample = df.sample(min(len(df), 2_000), random_state=42)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sample["square_feet"], sample[TARGET], alpha=0.35, s=16, color="#48639c")
        ax.set_title("Price vs. Interior Size")
        ax.set_xlabel("Square feet")
        ax.set_ylabel("Price (CAD)")
        ax.ticklabel_format(style="plain", axis="y")
        path = FIGURE_DIR / "price_vs_square_feet.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths.append(path)

    if "property_type" in df.columns:
        counts = df["property_type"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="barh", ax=ax, color="#5d737e")
        ax.set_title("Listings by Property Type")
        ax.set_xlabel("Listings")
        path = FIGURE_DIR / "property_type_counts.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths.append(path)

    return figure_paths


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline, list[str], list[str]]:
    feature_columns = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    if not feature_columns:
        raise ValueError("No usable feature columns were found.")

    numeric_features = [c for c in feature_columns if c in NUMERIC_FEATURES]
    categorical_features = [c for c in feature_columns if c in CATEGORICAL_FEATURES]

    X = df[feature_columns]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_specs = {
        "MedianBaseline": DummyRegressor(strategy="median"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=180,
            random_state=42,
            min_samples_leaf=5,
            n_jobs=-1,
        ),
    }

    rows = []
    fitted_random_forest: Pipeline | None = None
    for model_name, model in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rows.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": r2_score(y_test, predictions),
            }
        )
        if model_name == "RandomForest":
            fitted_random_forest = pipeline

    if fitted_random_forest is None:
        raise RuntimeError("RandomForest model was not fitted.")

    metrics = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return metrics, fitted_random_forest, numeric_features, categorical_features


def get_feature_names(
    fitted_pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    feature_names: list[str] = []

    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = categorical_pipeline.named_steps["onehot"]
        feature_names.extend(encoder.get_feature_names_out(categorical_features).tolist())

    return feature_names


def save_feature_importance(
    fitted_pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Path:
    model = fitted_pipeline.named_steps["model"]
    feature_names = get_feature_names(fitted_pipeline, numeric_features, categorical_features)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    output_path = REPORT_DIR / "feature_importance.csv"
    importance.to_csv(output_path, index=False)

    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#6b705c")
    ax.set_title("Top Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig_path = FIGURE_DIR / "feature_importance.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    return output_path


def format_metrics_table(metrics: pd.DataFrame) -> list[str]:
    lines = ["| Model | MAE | RMSE | R2 |", "| --- | ---: | ---: | ---: |"]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.model} | {row.mae:.3f} | {row.rmse:.3f} | {row.r2:.3f} |")
    return lines


def write_summary(
    input_path: Path,
    cleaning_report: dict[str, int],
    metrics: pd.DataFrame,
    figure_paths: list[Path],
) -> Path:
    best = metrics.iloc[0]
    lines = [
        "# Toronto Housing Price Analysis Summary",
        "",
        f"Input file: `{input_path}`",
        "",
        "## Cleaning",
        "",
    ]
    for key, value in cleaning_report.items():
        readable = key.replace("_", " ").capitalize()
        lines.append(f"- {readable}: {value:,}")

    lines.extend(
        [
            "",
            "## Model Results",
            "",
            *format_metrics_table(metrics),
            "",
            f"Best model by RMSE: `{best['model']}`",
            "",
            "## Figures",
            "",
        ]
    )
    for path in figure_paths:
        lines.append(f"- `{path}`")

    output_path = REPORT_DIR / "analysis_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def resolve_input_path(cli_input: str | None) -> Path:
    if cli_input:
        path = Path(cli_input)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    if RAW_MAIN_PATH.exists():
        return RAW_MAIN_PATH
    if RAW_SAMPLE_PATH.exists():
        return RAW_SAMPLE_PATH
    raise FileNotFoundError(
        "No input CSV found. Add data/raw/toronto_housing.csv or run "
        "`python -m toronto_housing_analysis.generate_sample_data` first."
    )


def run(input_path: Path) -> None:
    DATA_DIR.joinpath("processed").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_path)
    clean_df, cleaning_report = clean_data(raw_df)
    clean_df.to_csv(PROCESSED_PATH, index=False)

    figure_paths = make_figures(clean_df)
    metrics, fitted_random_forest, numeric_features, categorical_features = evaluate_models(clean_df)

    metrics_path = REPORT_DIR / "model_metrics.csv"
    metrics_json_path = REPORT_DIR / "model_metrics.json"
    metrics.to_csv(metrics_path, index=False)
    metrics_json_path.write_text(metrics.to_json(orient="records", indent=2), encoding="utf-8")

    save_feature_importance(fitted_random_forest, numeric_features, categorical_features)
    summary_path = write_summary(input_path, cleaning_report, metrics, figure_paths)

    print(f"Cleaned data: {PROCESSED_PATH}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Toronto housing price analysis.")
    parser.add_argument(
        "--input",
        help="Optional CSV path. Defaults to data/raw/toronto_housing.csv, then sample data.",
    )
    args = parser.parse_args()
    run(resolve_input_path(args.input))


if __name__ == "__main__":
    main()
