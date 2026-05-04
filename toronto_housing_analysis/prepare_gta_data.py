from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "greater-toronto-area-housing-data-master"
    / "greater-toronto-area-housing-data-master"
    / "data"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "toronto_housing.csv"


def parse_price(value: object) -> float | None:
    if pd.isna(value):
        return None
    cleaned = re.sub(r"[^0-9.]", "", str(value))
    return float(cleaned) if cleaned else None


def parse_bedrooms(details: object) -> float | None:
    if pd.isna(details):
        return None
    text = str(details).lower()
    if "studio" in text:
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*bds?\b", text)
    return float(match.group(1)) if match else None


def parse_bathrooms(details: object) -> float | None:
    if pd.isna(details):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*ba\b", str(details).lower())
    return float(match.group(1)) if match else None


def parse_square_feet(details: object) -> float | None:
    if pd.isna(details):
        return None
    match = re.search(r"([\d,]+)\s*sqft\b", str(details).lower())
    return float(match.group(1).replace(",", "")) if match else None


def parse_property_type(details: object) -> str:
    if pd.isna(details):
        return "Unknown"
    text = re.sub(r"\s+", " ", str(details).strip())
    match = re.search(r"-\s*(.*?)\s+for sale", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().title()
    if "condo" in text.lower():
        return "Condo"
    if "townhouse" in text.lower():
        return "Townhouse"
    if "house" in text.lower():
        return "House"
    return "Unknown"


def city_from_path(csv_path: Path) -> str:
    city = csv_path.parents[1].name
    return city.replace("_", " ").title()


def snapshot_date_from_filename(csv_path: Path) -> str | None:
    match = re.search(r"_(\d{8})\.json\.csv$", csv_path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%m%d%Y").date().isoformat()


def load_gta_csvs(source_root: Path = DEFAULT_SOURCE_ROOT) -> pd.DataFrame:
    csv_paths = sorted(source_root.glob("*/csv/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {source_root}")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame["city"] = city_from_path(csv_path)
        frame["neighborhood"] = frame["city"]
        frame["snapshot_date"] = snapshot_date_from_filename(csv_path)
        frame["source_file"] = csv_path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["price"] = df["price"].map(parse_price)
    df["bedrooms"] = df["details"].map(parse_bedrooms)
    df["bathrooms"] = df["details"].map(parse_bathrooms)
    df["square_feet"] = df["details"].map(parse_square_feet)
    df["property_type"] = df["details"].map(parse_property_type)

    return df[
        [
            "address",
            "price",
            "bedrooms",
            "bathrooms",
            "square_feet",
            "property_type",
            "city",
            "neighborhood",
            "snapshot_date",
            "details",
            "source_file",
        ]
    ]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_gta_csvs()
    df.to_csv(OUTPUT_PATH, index=False)
    unique_listings = df[["address", "price", "details"]].drop_duplicates().shape[0]
    print(f"Wrote {len(df):,} listing snapshots to {OUTPUT_PATH}")
    print(f"Unique address/price/detail combinations: {unique_listings:,}")


if __name__ == "__main__":
    main()
