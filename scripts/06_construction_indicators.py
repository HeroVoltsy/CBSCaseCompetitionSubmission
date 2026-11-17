
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/processed")
OUT = RAW / "market_indicators"
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    return pd.read_csv(RAW / name, sep=",", skipinitialspace=True)

bygv01 = load("bygv01.csv")
bygv11 = load("bygv11.csv")
bygb34 = load("bygb34.csv")
bygb40 = load("bygb40.csv")
byg43 = load("byg43.csv")
pris91 = load("pris91.csv")

macro = pd.read_csv(RAW / "macroeconomic/denmark_macro_annual.csv")

for df in [bygv01, bygv11, bygb34, bygb40, byg43, pris91]:
    df.columns = df.columns.str.strip()
    df["INDHOLD"] = pd.to_numeric(df["INDHOLD"], errors="coerce")

bygv01["TID"] = pd.to_numeric(bygv01["TID"], errors="coerce")
bygv01["BYGFASE"] = bygv01["BYGFASE"].str.strip()
bygv01["ANVEND"] = bygv01["ANVEND"].str.strip()

annual = (
    bygv01.groupby(["TID", "BYGFASE"], as_index=False)["INDHOLD"]
    .sum()
    .pivot(index="TID", columns="BYGFASE", values="INDHOLD")
    .reset_index()
)
annual.to_csv(OUT / "construction_activity_annual.csv", index=False)

use_activity = (
    bygv01.groupby(["TID", "ANVEND", "BYGFASE"], as_index=False)["INDHOLD"]
    .sum()
)
use_activity.to_csv(OUT / "construction_by_use.csv", index=False)

quarterly = bygv11.copy()
quarterly["TID"] = quarterly["TID"].astype(str).str.strip()
quarterly = quarterly[quarterly["OMRÅDE"].str.strip() == "All Denmark"]
quarterly = (
    quarterly.groupby(["TID", "BYGFASE"], as_index=False)["INDHOLD"]
    .sum()
)
quarterly.to_csv(OUT / "construction_activity_quarterly.csv", index=False)

stock = bygb34.copy()
stock["TID"] = pd.to_numeric(stock["TID"], errors="coerce")
stock = stock[stock["OMRÅDE"].str.strip() == "All Denmark"]

stock_age = (
    stock.groupby(["TID", "OPFØRELSESÅR"], as_index=False)["INDHOLD"]
    .sum()
)
stock_age.to_csv(OUT / "building_stock_by_age.csv", index=False)

stock_use = (
    stock.groupby(["TID", "ANVEND"], as_index=False)["INDHOLD"]
    .sum()
)
stock_use.to_csv(OUT / "building_stock_by_use.csv", index=False)

cost = byg43.copy()
cost["TID"] = cost["TID"].astype(str).str.strip()
cost = cost[
    cost["DINDEKS"].str.contains("total", case=False, na=False)
    & cost["TAL"].str.strip().eq("Index")
]
cost.to_csv(OUT / "construction_cost_index.csv", index=False)

renovation = pris91.copy()
renovation["TID"] = pd.to_numeric(renovation["TID"], errors="coerce")
renovation = renovation[
    renovation["ARBART"].str.contains("total", case=False, na=False)
]
renovation.to_csv(OUT / "renovation_cost_index.csv", index=False)

macro["TID"] = pd.to_numeric(macro["TID"], errors="coerce")
annual["TID"] = pd.to_numeric(annual["TID"], errors="coerce")

combined = annual.merge(macro, on="TID", how="left")
combined.to_csv(OUT / "construction_macro_panel.csv", index=False)

print("Construction indicators saved:", OUT)
for f in sorted(OUT.glob("*.csv")):
    df = pd.read_csv(f)
    print(f"{f.name}: {df.shape}")
