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
PROCESSED_PATH = DATA_DIR / "processed" / "cleaned_toronto_housing.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "price"
NUMERIC_FEATURES = ["bedrooms", "bathrooms", "square_feet", "parking", "days_on_market"]
CATEGORICAL_FEATURES = ["property_type", "city", "neighborhood", "postal_fsa"]
PRICE_TIER_BINS = [0, 600_000, 900_000, 1_200_000, 1_800_000, np.inf]
PRICE_TIER_LABELS = ["Under 600K", "600K-900K", "900K-1.2M", "1.2M-1.8M", "1.8M+"]
MIN_REASONABLE_PRICE = 100_000

COLUMN_ALIASES = {
    "price": ["price", "sold_price", "sale_price", "selling_price", "list_price", "asking_price"],
    "bedrooms": ["bedrooms", "beds", "bed", "br"],
    "bathrooms": ["bathrooms", "baths", "bath", "ba"],
    "square_feet": ["square_feet", "sqft", "sq_ft", "size_sqft", "living_area", "area_sqft"],
    "property_type": ["property_type", "home_type", "house_type", "type", "propertytype"],
    "city": ["city", "municipality", "town"],
    "neighborhood": ["neighborhood", "neighbourhood", "community", "district", "area"],
    "postal_fsa": ["postal_fsa", "fsa", "postal_area"],
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


def add_price_tiers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_tier"] = pd.cut(
        df[TARGET],
        bins=PRICE_TIER_BINS,
        labels=PRICE_TIER_LABELS,
        include_lowest=True,
        right=False,
    )
    return df


def location_column(df: pd.DataFrame) -> str | None:
    if "city" in df.columns:
        return "city"
    if "neighborhood" in df.columns:
        return "neighborhood"
    return None


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

    low_price_rows = int((df[TARGET] < MIN_REASONABLE_PRICE).sum())
    df = df[df[TARGET] >= MIN_REASONABLE_PRICE].copy()
    report["low_price_rows_removed"] = low_price_rows

    for column in [c for c in CATEGORICAL_FEATURES if c in df.columns]:
        if column == "property_type":
            df[column] = df[column].map(standardize_property_type)
        else:
            df[column] = df[column].map(standardize_category)

    missing_feature_values = int(df[[c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]].isna().sum().sum())
    report["missing_feature_values_found"] = missing_feature_values
    for column in [c for c in CATEGORICAL_FEATURES if c in df.columns]:
        df[column] = df[column].fillna("Unknown")

    df, removed_price_outliers = remove_iqr_outliers(df, TARGET, multiplier=1.75)
    df, removed_size_outliers = remove_iqr_outliers(df, "square_feet", multiplier=2.0)
    report["price_outliers_removed"] = removed_price_outliers
    report["size_outliers_removed"] = removed_size_outliers
    report["rows_after_cleaning"] = len(df)
    df = add_price_tiers(df)

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

    location = location_column(df)
    if location:
        medians = df.groupby(location)[TARGET].median().sort_values().tail(12)
        fig, ax = plt.subplots(figsize=(9, 6))
        medians.plot(kind="barh", ax=ax, color="#8c5e3c")
        ax.set_title(f"Median Price by {location.replace('_', ' ').title()}")
        ax.set_xlabel("Median price (CAD)")
        ax.ticklabel_format(style="plain", axis="x")
        path = FIGURE_DIR / f"median_price_by_{location}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_paths.append(path)

    if "square_feet" in df.columns:
        size_df = df.dropna(subset=["square_feet", TARGET])
        if not size_df.empty:
            sample = size_df.sample(min(len(size_df), 2_000), random_state=42)
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


def summary_aggregations(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    aggregations: dict[str, tuple[str, str]] = {
        "listing_count": (TARGET, "size"),
        "median_price": (TARGET, "median"),
        "average_price": (TARGET, "mean"),
        "min_price": (TARGET, "min"),
        "max_price": (TARGET, "max"),
    }
    if "bedrooms" in df.columns:
        aggregations["median_bedrooms"] = ("bedrooms", "median")
    if "bathrooms" in df.columns:
        aggregations["median_bathrooms"] = ("bathrooms", "median")
    if "square_feet" in df.columns:
        aggregations["median_square_feet"] = ("square_feet", "median")
    return aggregations


def save_city_market_analysis(df: pd.DataFrame) -> list[Path]:
    location = location_column(df)
    if not location:
        return []

    summary = (
        df.groupby(location)
        .agg(**summary_aggregations(df))
        .sort_values("median_price", ascending=False)
        .reset_index()
    )
    output_path = REPORT_DIR / "city_market_summary.csv"
    summary.to_csv(output_path, index=False)

    top = summary.head(12).sort_values("median_price")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top[location], top["median_price"], color="#3c6e71")
    ax.set_title(f"Median Listing Price by {location.replace('_', ' ').title()}")
    ax.set_xlabel("Median price (CAD)")
    ax.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    fig_path = FIGURE_DIR / "analysis_city_median_price.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    return [output_path, fig_path]


def save_property_type_analysis(df: pd.DataFrame) -> list[Path]:
    if "property_type" not in df.columns:
        return []

    summary = (
        df.groupby("property_type")
        .agg(**summary_aggregations(df))
        .sort_values("median_price", ascending=False)
        .reset_index()
    )
    output_path = REPORT_DIR / "property_type_summary.csv"
    summary.to_csv(output_path, index=False)

    top = summary.sort_values("median_price")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["property_type"], top["median_price"], color="#7f5539")
    ax.set_title("Median Listing Price by Property Type")
    ax.set_xlabel("Median price (CAD)")
    ax.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    fig_path = FIGURE_DIR / "analysis_property_type_median_price.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    return [output_path, fig_path]


def save_bedroom_bathroom_analysis(df: pd.DataFrame) -> list[Path]:
    output_paths: list[Path] = []
    if "bedrooms" in df.columns:
        bedroom_df = df.dropna(subset=["bedrooms"]).copy()
        bedroom_df = bedroom_df[bedroom_df["bedrooms"].between(0, 8)]
        if not bedroom_df.empty:
            bedroom_summary = (
                bedroom_df.groupby("bedrooms")
                .agg(**summary_aggregations(bedroom_df))
                .sort_index()
                .reset_index()
            )
            bedroom_path = REPORT_DIR / "bedroom_price_summary.csv"
            bedroom_summary.to_csv(bedroom_path, index=False)
            output_paths.append(bedroom_path)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                bedroom_summary["bedrooms"],
                bedroom_summary["median_price"],
                marker="o",
                color="#4a5759",
            )
            ax.set_title("Median Listing Price by Bedroom Count")
            ax.set_xlabel("Bedrooms")
            ax.set_ylabel("Median price (CAD)")
            ax.ticklabel_format(style="plain", axis="y")
            fig.tight_layout()
            fig_path = FIGURE_DIR / "analysis_bedroom_median_price.png"
            fig.savefig(fig_path, dpi=160)
            plt.close(fig)
            output_paths.append(fig_path)

    if {"bedrooms", "bathrooms"}.issubset(df.columns):
        matrix_df = df.dropna(subset=["bedrooms", "bathrooms"]).copy()
        matrix_df = matrix_df[
            matrix_df["bedrooms"].between(0, 8) & matrix_df["bathrooms"].between(1, 8)
        ]
        if not matrix_df.empty:
            matrix = matrix_df.pivot_table(
                index="bedrooms",
                columns="bathrooms",
                values=TARGET,
                aggfunc="median",
            )
            matrix_path = REPORT_DIR / "bedroom_bathroom_price_matrix.csv"
            matrix.to_csv(matrix_path)
            output_paths.append(matrix_path)

    return output_paths


def save_price_tier_analysis(df: pd.DataFrame) -> list[Path]:
    if "price_tier" not in df.columns:
        return []

    output_paths: list[Path] = []
    tier_summary = (
        df.groupby("price_tier", observed=False)
        .agg(**summary_aggregations(df))
        .reset_index()
    )
    tier_path = REPORT_DIR / "price_tier_summary.csv"
    tier_summary.to_csv(tier_path, index=False)
    output_paths.append(tier_path)

    location = location_column(df)
    if location:
        tier_counts = pd.crosstab(df[location], df["price_tier"])
        tier_counts_path = REPORT_DIR / "price_tier_by_city.csv"
        tier_counts.to_csv(tier_counts_path)
        output_paths.append(tier_counts_path)

        top_locations = df[location].value_counts().head(10).index
        plot_counts = tier_counts.loc[tier_counts.index.intersection(top_locations)]
        plot_counts = plot_counts.loc[df[location].value_counts().loc[plot_counts.index].sort_values().index]
        if not plot_counts.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_counts.plot(kind="barh", stacked=True, ax=ax, colormap="viridis")
            ax.set_title("Price Tier Mix by City")
            ax.set_xlabel("Listings")
            ax.set_ylabel(location.replace("_", " ").title())
            ax.legend(title="Price tier", bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            fig_path = FIGURE_DIR / "analysis_price_tier_by_city.png"
            fig.savefig(fig_path, dpi=160)
            plt.close(fig)
            output_paths.append(fig_path)

    return output_paths


def save_repeated_listing_analysis(df: pd.DataFrame) -> list[Path]:
    required = {"address", "snapshot_date", TARGET}
    if not required.issubset(df.columns):
        return []

    records = df.dropna(subset=["address", "snapshot_date", TARGET]).copy()
    if records.empty:
        return []

    records["snapshot_date"] = pd.to_datetime(records["snapshot_date"], errors="coerce")
    records = records.dropna(subset=["snapshot_date"])
    records = records.sort_values(["address", "snapshot_date"])
    records = records.drop_duplicates(subset=["address", "snapshot_date", TARGET])

    if records.empty:
        return []

    group_columns = {
        "observation_count": ("snapshot_date", "count"),
        "days_seen": ("snapshot_date", "nunique"),
        "first_snapshot": ("snapshot_date", "min"),
        "last_snapshot": ("snapshot_date", "max"),
        "min_price": (TARGET, "min"),
        "max_price": (TARGET, "max"),
        "price_levels": (TARGET, "nunique"),
    }
    if "city" in records.columns:
        group_columns["city"] = ("city", "first")
    if "property_type" in records.columns:
        group_columns["property_type"] = ("property_type", "first")
    if "bedrooms" in records.columns:
        group_columns["bedrooms"] = ("bedrooms", "first")
    if "bathrooms" in records.columns:
        group_columns["bathrooms"] = ("bathrooms", "first")

    summary = records.groupby("address").agg(**group_columns).reset_index()
    first_prices = records.drop_duplicates("address", keep="first")[["address", TARGET]]
    last_prices = records.drop_duplicates("address", keep="last")[["address", TARGET]]
    summary = summary.merge(first_prices.rename(columns={TARGET: "first_price"}), on="address")
    summary = summary.merge(last_prices.rename(columns={TARGET: "last_price"}), on="address")
    summary["price_change"] = summary["last_price"] - summary["first_price"]
    summary["price_change_pct"] = summary["price_change"] / summary["first_price"]
    summary = summary.sort_values(["days_seen", "price_levels", "max_price"], ascending=False)

    repeated_path = REPORT_DIR / "repeated_listing_summary.csv"
    summary.to_csv(repeated_path, index=False)

    changes = summary[(summary["days_seen"] > 1) & (summary["price_change"] != 0)].copy()
    changes = changes.sort_values("price_change", key=lambda s: s.abs(), ascending=False)
    changes_path = REPORT_DIR / "listing_price_changes.csv"
    changes.to_csv(changes_path, index=False)

    output_paths = [repeated_path, changes_path]

    observation_counts = summary["days_seen"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(observation_counts.index.astype(str), observation_counts.values, color="#577590")
    ax.set_title("Listing Reappearances Across Daily Snapshots")
    ax.set_xlabel("Days seen")
    ax.set_ylabel("Listings")
    fig.tight_layout()
    fig_path = FIGURE_DIR / "analysis_repeated_listing_observations.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    output_paths.append(fig_path)

    if not changes.empty:
        top_changes = changes.head(12).copy()
        top_changes["label"] = top_changes["address"].str.slice(0, 38)
        top_changes = top_changes.sort_values("price_change")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = np.where(top_changes["price_change"] >= 0, "#2a9d8f", "#b56576")
        ax.barh(top_changes["label"], top_changes["price_change"], color=colors)
        ax.set_title("Largest Listing Price Changes")
        ax.set_xlabel("Price change (CAD)")
        ax.ticklabel_format(style="plain", axis="x")
        fig.tight_layout()
        fig_path = FIGURE_DIR / "analysis_largest_price_changes.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        output_paths.append(fig_path)

    return output_paths


def run_key_analyses(df: pd.DataFrame) -> list[Path]:
    analysis_paths: list[Path] = []
    analysis_paths.extend(save_city_market_analysis(df))
    analysis_paths.extend(save_property_type_analysis(df))
    analysis_paths.extend(save_bedroom_bathroom_analysis(df))
    analysis_paths.extend(save_price_tier_analysis(df))
    analysis_paths.extend(save_repeated_listing_analysis(df))
    return analysis_paths


def model_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical_features = []

    if "property_type" in df.columns:
        categorical_features.append("property_type")

    location = location_column(df)
    if location:
        categorical_features.append(location)

    if "postal_fsa" in df.columns:
        categorical_features.append("postal_fsa")

    return numeric_features, categorical_features


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
    numeric_features, categorical_features = model_feature_lists(df)
    feature_columns = numeric_features + categorical_features
    if not feature_columns:
        raise ValueError("No usable feature columns were found.")

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
    names = preprocessor.get_feature_names_out()
    return [name.split("__", 1)[1] if "__" in name else name for name in names]


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
    analysis_paths: list[Path],
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
            "## Key Analysis Outputs",
            "",
        ]
    )
    for path in analysis_paths:
        lines.append(f"- `{path}`")

    lines.extend(
        [
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
    raise FileNotFoundError(
        "No input CSV found. Run `python -m toronto_housing_analysis.prepare_gta_data` "
        "or add data/raw/toronto_housing.csv."
    )


def run(input_path: Path) -> None:
    DATA_DIR.joinpath("processed").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_path)
    clean_df, cleaning_report = clean_data(raw_df)
    clean_df.to_csv(PROCESSED_PATH, index=False)

    figure_paths = make_figures(clean_df)
    analysis_paths = run_key_analyses(clean_df)
    metrics, fitted_random_forest, numeric_features, categorical_features = evaluate_models(clean_df)

    metrics_path = REPORT_DIR / "model_metrics.csv"
    metrics_json_path = REPORT_DIR / "model_metrics.json"
    metrics.to_csv(metrics_path, index=False)
    metrics_json_path.write_text(metrics.to_json(orient="records", indent=2), encoding="utf-8")

    save_feature_importance(fitted_random_forest, numeric_features, categorical_features)
    summary_path = write_summary(input_path, cleaning_report, metrics, figure_paths, analysis_paths)

    print(f"Cleaned data: {PROCESSED_PATH}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    print(metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Toronto housing price analysis.")
    parser.add_argument(
        "--input",
        help="Optional CSV path. Defaults to data/raw/toronto_housing.csv.",
    )
    args = parser.parse_args()
    run(resolve_input_path(args.input))


if __name__ == "__main__":
    main()
