
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("data/processed")
OUT = Path("outputs")

FIGURES = OUT / "figures"
TABLES = OUT / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

def load(name):
    df = pd.read_csv(BASE / f"{name}.csv")
    df["INDHOLD"] = pd.to_numeric(df["INDHOLD"], errors="coerce")
    return df

def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close()

# 1. Construction activity: annual
df = load("bygv01")

df["YEAR"] = pd.to_numeric(df["TID"], errors="coerce")

annual = (
    df.groupby(["YEAR", "BYGFASE", "ANVEND"], as_index=False)
    ["INDHOLD"].sum()
)

annual.to_csv(TABLES / "annual_construction.csv", index=False)

residential = [
    "Farm houses",
    "Detached houses",
    "Terraced, linked or semi-detached houses",
    "Multi-dwelling houses",
    "Student hostels",
    "Residential buildings for communities",
    "Other residential building"
]

df["SEGMENT"] = df["ANVEND"].apply(
    lambda x: "Residential" if x in residential else "Non-residential"
)

segment = (
    df.groupby(["YEAR", "BYGFASE", "SEGMENT"], as_index=False)
    ["INDHOLD"].sum()
)

segment.to_csv(TABLES / "construction_segments.csv", index=False)

for phase in ["Started", "Completed"]:
    subset = segment[segment["BYGFASE"] == phase]

    pivot = subset.pivot(
        index="YEAR",
        columns="SEGMENT",
        values="INDHOLD"
    )

    pivot.plot(figsize=(11, 5), marker="o")
    plt.title(f"Annual {phase} Construction")
    plt.xlabel("Year")
    plt.ylabel("Floor area (m²)")
    plt.grid(alpha=0.3)
    save_plot(f"annual_{phase.lower()}_construction")

# 2. Quarterly construction activity
df = load("bygv11")

df["QUARTER"] = pd.PeriodIndex(
    df["TID"].astype(str).str.replace("Q", "Q"),
    freq="Q"
)

quarterly = (
    df.groupby(["QUARTER", "BYGFASE", "ANVEND"], as_index=False)
    ["INDHOLD"].sum()
)

quarterly.to_csv(TABLES / "quarterly_construction.csv", index=False)

quarterly_segments = (
    df.assign(
        SEGMENT=df["ANVEND"].apply(
            lambda x: "Residential" if x in residential else "Non-residential"
        )
    )
    .groupby(["QUARTER", "BYGFASE", "SEGMENT"], as_index=False)
    ["INDHOLD"].sum()
)

for phase in ["Started", "Completed"]:
    subset = quarterly_segments[
        quarterly_segments["BYGFASE"] == phase
    ]

    pivot = subset.pivot(
        index="QUARTER",
        columns="SEGMENT",
        values="INDHOLD"
    )

    pivot.plot(figsize=(12, 5))
    plt.title(f"Quarterly {phase} Construction")
    plt.xlabel("Quarter")
    plt.ylabel("Floor area (m²)")
    plt.grid(alpha=0.3)
    save_plot(f"quarterly_{phase.lower()}_construction")

# 3. Existing building stock
df = load("bygb34")

df["YEAR"] = pd.to_numeric(df["TID"], errors="coerce")

stock = (
    df[df["AREALTYPE"] == "Floorspace - above basement"]
    .groupby(["YEAR", "OPFØRELSESÅR", "ANVEND"], as_index=False)
    ["INDHOLD"].sum()
)

stock.to_csv(TABLES / "building_stock.csv", index=False)

age_stock = (
    stock.groupby(["YEAR", "OPFØRELSESÅR"], as_index=False)
    ["INDHOLD"].sum()
)

latest_year = age_stock["YEAR"].max()

latest_stock = age_stock[
    age_stock["YEAR"] == latest_year
].sort_values("INDHOLD", ascending=False)

latest_stock.to_csv(TABLES / "building_stock_age_latest.csv", index=False)

plt.figure(figsize=(12, 6))
plt.bar(
    latest_stock["OPFØRELSESÅR"],
    latest_stock["INDHOLD"]
)
plt.title(f"Building Stock by Construction Period ({int(latest_year)})")
plt.xlabel("Construction period")
plt.ylabel("Floor area (1,000 m²)")
plt.xticks(rotation=70)
plt.grid(axis="y", alpha=0.3)
save_plot("building_stock_by_age")

# 4. Heated area and heating types
df = load("bygb40")

df["YEAR"] = pd.to_numeric(df["TID"], errors="coerce")

heating = (
    df.groupby(["YEAR", "OPVARM"], as_index=False)
    ["INDHOLD"].sum()
)

heating.to_csv(TABLES / "heated_area_by_type.csv", index=False)

latest_heating = heating[
    heating["YEAR"] == heating["YEAR"].max()
]

latest_heating = latest_heating.sort_values(
    "INDHOLD", ascending=False
)

latest_heating.to_csv(
    TABLES / "latest_heating_distribution.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.barh(
    latest_heating["OPVARM"],
    latest_heating["INDHOLD"]
)
plt.title("Heated Building Area by Heating Type")
plt.xlabel("Heated area (1,000 m²)")
plt.grid(axis="x", alpha=0.3)
save_plot("heated_area_by_heating_type")

# 5. Construction cost indices
df = load("byg43")

cost = df[
    (df["HINDEKS"] == "Construction Cost Index for residential buildings") &
    (df["DINDEKS"] == "Construction Cost Index, total") &
    (df["TAL"] == "Index")
].copy()

cost_pivot = cost.pivot(
    index="TID",
    columns="ART",
    values="INDHOLD"
)

cost_pivot.to_csv(TABLES / "construction_cost_indices.csv")

cost_pivot.plot(figsize=(11, 5), marker=".")
plt.title("Residential Construction Cost Indices")
plt.xlabel("Quarter")
plt.ylabel("Index")
plt.grid(alpha=0.3)
save_plot("construction_cost_indices")

# 6. Renovation price indices
df = load("pris91")

renovation = df[df["TAL"] == "Index"].copy()

renovation_pivot = renovation.pivot(
    index="TID",
    columns="ARBART",
    values="INDHOLD"
)

renovation_pivot.to_csv(TABLES / "renovation_price_indices.csv")

renovation_pivot.plot(figsize=(11, 5), marker="o")
plt.title("Renovation and Maintenance Price Indices")
plt.xlabel("Year")
plt.ylabel("Index")
plt.grid(alpha=0.3)
save_plot("renovation_price_indices")

# 7. Basic dataset diagnostics
datasets = [
    "bygv01",
    "bygv11",
    "bygb34",
    "bygb40",
    "byg43",
    "pris91"
]

diagnostics = []

for name in datasets:
    df = load(name)

    diagnostics.append({
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_periods": df["TID"].nunique()
    })

pd.DataFrame(diagnostics).to_csv(
    TABLES / "data_diagnostics.csv",
    index=False
)

print("\nMARKET ANALYSIS COMPLETE")
print(pd.DataFrame(diagnostics).to_string(index=False))
print(f"\nFigures: {FIGURES}")
print(f"Tables: {TABLES}")
