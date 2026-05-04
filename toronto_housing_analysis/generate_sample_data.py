from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "toronto_housing_sample.csv"


NEIGHBORHOODS = {
    "Downtown Core": 980_000,
    "North York": 875_000,
    "Scarborough": 760_000,
    "Etobicoke": 820_000,
    "York": 790_000,
    "East York": 850_000,
    "The Beaches": 1_080_000,
    "High Park": 1_000_000,
    "Liberty Village": 825_000,
    "Roncesvalles": 1_020_000,
    "The Annex": 1_120_000,
    "Leslieville": 960_000,
}

PROPERTY_TYPES = {
    "Condo": {"base_size": 690, "price_multiplier": 0.82},
    "Townhouse": {"base_size": 1_250, "price_multiplier": 0.98},
    "Semi-Detached": {"base_size": 1_550, "price_multiplier": 1.12},
    "Detached": {"base_size": 2_050, "price_multiplier": 1.35},
}

MESSY_PROPERTY_LABELS = {
    "Condo": ["Condo", "condo", "Condo Apartment", "apartment condo"],
    "Townhouse": ["Townhouse", "townhome", "Town House"],
    "Semi-Detached": ["Semi-Detached", "semi detached", "Semi Detached Home"],
    "Detached": ["Detached", "detached home", "Detached House"],
}


def make_sample_data(rows: int = 6_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    neighborhoods = np.array(list(NEIGHBORHOODS))
    property_types = np.array(list(PROPERTY_TYPES))

    chosen_neighborhoods = rng.choice(
        neighborhoods,
        size=rows,
        p=np.array([0.15, 0.12, 0.13, 0.1, 0.06, 0.07, 0.06, 0.08, 0.08, 0.05, 0.05, 0.05]),
    )
    chosen_property_types = rng.choice(property_types, size=rows, p=[0.46, 0.19, 0.15, 0.20])

    bedrooms = np.array(
        [
            rng.choice([1, 2, 3, 4, 5], p=[0.3, 0.36, 0.2, 0.1, 0.04])
            if property_type == "Condo"
            else rng.choice([2, 3, 4, 5, 6], p=[0.16, 0.35, 0.3, 0.14, 0.05])
            for property_type in chosen_property_types
        ]
    )
    bathrooms = np.maximum(1, np.round(bedrooms * rng.normal(0.62, 0.13, rows))).astype(float)
    parking = np.array(
        [
            rng.choice([0, 1, 2], p=[0.55, 0.4, 0.05])
            if property_type == "Condo"
            else rng.choice([0, 1, 2, 3], p=[0.08, 0.38, 0.42, 0.12])
            for property_type in chosen_property_types
        ],
        dtype=float,
    )
    days_on_market = np.maximum(1, rng.gamma(shape=2.1, scale=11, size=rows)).round().astype(int)

    square_feet = []
    for property_type, bed_count in zip(chosen_property_types, bedrooms):
        base_size = PROPERTY_TYPES[property_type]["base_size"]
        size = base_size + (bed_count - 2) * rng.normal(185, 35) + rng.normal(0, 165)
        square_feet.append(np.clip(size, 380, 4_800))
    square_feet = np.array(square_feet).round().astype(float)

    prices = []
    for neighborhood, property_type, bed_count, bath_count, sqft, park_count, dom in zip(
        chosen_neighborhoods,
        chosen_property_types,
        bedrooms,
        bathrooms,
        square_feet,
        parking,
        days_on_market,
    ):
        neighborhood_base = NEIGHBORHOODS[neighborhood]
        property_multiplier = PROPERTY_TYPES[property_type]["price_multiplier"]
        modeled_price = (
            neighborhood_base * property_multiplier
            + sqft * rng.normal(350, 45)
            + bed_count * 22_000
            + bath_count * 18_500
            + park_count * 31_000
            - dom * 1_150
            + rng.normal(0, 95_000)
        )
        prices.append(max(250_000, modeled_price))
    prices = np.array(prices).round(-3)

    messy_property_types = [
        rng.choice(MESSY_PROPERTY_LABELS[property_type]) for property_type in chosen_property_types
    ]
    messy_neighborhoods = [
        rng.choice([name, name.lower(), name.upper(), f" {name} "]) for name in chosen_neighborhoods
    ]

    listing_dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=rows), unit="D"
    )

    df = pd.DataFrame(
        {
            "price": prices,
            "bedrooms": bedrooms.astype(float),
            "bathrooms": bathrooms,
            "square_feet": square_feet,
            "property_type": messy_property_types,
            "neighborhood": messy_neighborhoods,
            "parking": parking,
            "days_on_market": days_on_market,
            "listing_date": listing_dates.strftime("%Y-%m-%d"),
        }
    )

    missing_specs = {
        "bathrooms": 0.025,
        "square_feet": 0.035,
        "property_type": 0.018,
        "neighborhood": 0.012,
        "parking": 0.055,
    }
    for column, rate in missing_specs.items():
        mask = rng.random(rows) < rate
        df.loc[mask, column] = np.nan

    high_outliers = rng.choice(df.index, size=20, replace=False)
    low_outliers = rng.choice(df.index.difference(high_outliers), size=15, replace=False)
    df.loc[high_outliers, "price"] = df.loc[high_outliers, "price"] * rng.uniform(3.2, 5.5, size=20)
    df.loc[low_outliers, "price"] = df.loc[low_outliers, "price"] * rng.uniform(0.18, 0.35, size=15)

    duplicate_rows = df.sample(35, random_state=seed)
    return pd.concat([df, duplicate_rows], ignore_index=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = make_sample_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
