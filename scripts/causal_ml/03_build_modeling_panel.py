
import pandas as pd
import numpy as np
from pathlib import Path

base = Path("data/processed/market_indicators")
macro_path = Path("data/processed/macroeconomic/denmark_macro_annual.csv")
output = Path("data/processed/causal_ml")
output.mkdir(parents=True, exist_ok=True)

by_use = pd.read_csv(base / "construction_by_use.csv")
macro = pd.read_csv(macro_path)

panel = (
    by_use.pivot_table(
        index=["TID", "ANVEND"],
        columns="BYGFASE",
        values="INDHOLD",
        aggfunc="sum"
    )
    .reset_index()
)
panel.columns.name = None

phases = ["Started", "Permits", "Completed", "Under construction"]

for col in phases:
    panel[f"{col}_lag1"] = panel.groupby("ANVEND")[col].shift(1)

panel["Started_lag2"] = panel.groupby("ANVEND")["Started"].shift(2)
panel["Started_growth_lag1"] = (
    panel.groupby("ANVEND")["Started"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    .groupby(panel["ANVEND"])
    .shift(1)
)

macro = macro.sort_values("TID").copy()
macro["gdp_growth_pct"] = macro["real_gdp_2020_chained"].pct_change() * 100
macro["cpi_growth_pct"] = macro["cpi"].pct_change() * 100

macro_features = [
    "TID",
    "gdp_growth_pct",
    "inflation_pct",
    "corporate_lending_rate_pct",
    "household_lending_rate_pct"
]

macro_features = macro[macro_features].copy()

for col in macro_features.columns:
    if col != "TID":
        macro_features[f"{col}_lag1"] = macro_features[col].shift(1)

panel = panel.merge(
    macro_features[["TID"] + [
        "gdp_growth_pct_lag1",
        "inflation_pct_lag1",
        "corporate_lending_rate_pct_lag1",
        "household_lending_rate_pct_lag1"
    ]],
    on="TID",
    how="left"
)

annual = pd.read_csv(base / "construction_activity_annual.csv")
reconciliation = (
    panel.groupby("TID")[phases]
    .sum()
    .reset_index()
    .merge(annual, on="TID", suffixes=("_panel", "_national"))
)

print("\nPANEL RECONCILIATION")
for phase in phases:
    panel_col = f"{phase}_panel"
    national_col = f"{phase}_national"
    reconciliation[f"{phase}_difference"] = (
        reconciliation[panel_col] - reconciliation[national_col]
    )
    print(
        phase,
        "maximum absolute difference:",
        reconciliation[f"{phase}_difference"].abs().max()
    )

model_features = [
    "ANVEND",
    "Started_lag1",
    "Started_lag2",
    "Permits_lag1",
    "Completed_lag1",
    "Under construction_lag1"
]

macro_model_features = model_features + [
    "gdp_growth_pct_lag1",
    "inflation_pct_lag1",
    "corporate_lending_rate_pct_lag1",
    "household_lending_rate_pct_lag1"
]

full_model = panel[
    ["TID", "ANVEND", "Started"] + model_features[1:]
].dropna().copy()

macro_model = panel[
    ["TID", "ANVEND", "Started"] + macro_model_features[1:]
].dropna().copy()

full_model.to_csv(output / "construction_panel_modeling.csv", index=False)
macro_model.to_csv(output / "construction_panel_macro_modeling.csv", index=False)
reconciliation.to_csv(output / "construction_panel_reconciliation.csv", index=False)

print("\nFULL-HISTORY MODEL")
print("Shape:", full_model.shape)
print("Years:", full_model["TID"].min(), "-", full_model["TID"].max())
print("Categories:", full_model["ANVEND"].nunique())
print("Missing:", full_model.isna().sum().sum())

print("\nMACRO MODEL")
print("Shape:", macro_model.shape)
print("Years:", macro_model["TID"].min(), "-", macro_model["TID"].max())
print("Categories:", macro_model["ANVEND"].nunique())
print("Missing:", macro_model.isna().sum().sum())

print("\nLATEST OBSERVATIONS")
print(
    full_model.sort_values(["ANVEND", "TID"])
    .groupby("ANVEND")
    .tail(1)[["TID", "ANVEND", "Started"]]
    .to_string(index=False)
)
