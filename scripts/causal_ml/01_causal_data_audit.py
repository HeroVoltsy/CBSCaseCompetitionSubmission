
import pandas as pd
from pathlib import Path

input_dir = Path("data/processed/market_indicators")

files = [
    "construction_activity_annual.csv",
    "construction_activity_quarterly.csv",
    "construction_by_use.csv",
    "building_stock_by_age.csv",
    "building_stock_by_use.csv",
    "construction_cost_index.csv",
    "renovation_cost_index.csv",
    "construction_macro_panel.csv"
]

for filename in files:
    path = input_dir / filename

    print(f"\n{'=' * 90}")
    print(filename)

    if not path.exists():
        print("NOT FOUND")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nUnique values:")
    for col in df.columns:
        if df[col].nunique() <= 20:
            print(f"{col}: {df[col].dropna().unique().tolist()}")

    print("\nSample:")
    print(df.head(5).to_string(index=False))

    print("\nNumeric summary:")
    print(df.describe().round(3).to_string())
