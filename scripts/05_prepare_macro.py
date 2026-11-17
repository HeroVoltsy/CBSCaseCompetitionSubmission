import pandas as pd
from pathlib import Path

RAW = Path("data/raw/macroeconomic")
OUT = Path("data/processed/macroeconomic")
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    return pd.read_csv(RAW / f"{name}.csv", sep=";")

gdp = load("gdp_real")
cpi = load("cpi")
inflation = load("inflation")
rates = load("interest_rates")

for df in [gdp, cpi, inflation]:
    df["TID"] = df["TID"].astype(int)
    df["INDHOLD"] = pd.to_numeric(df["INDHOLD"], errors="coerce")

gdp = gdp[["TID", "INDHOLD"]].rename(columns={"INDHOLD": "real_gdp_2020_chained"})
cpi = cpi[["TID", "INDHOLD"]].rename(columns={"INDHOLD": "cpi"})
inflation = inflation[["TID", "INDHOLD"]].rename(columns={"INDHOLD": "inflation_pct"})

macro = gdp.merge(cpi, on="TID", how="outer")
macro = macro.merge(inflation, on="TID", how="outer")
macro = macro.sort_values("TID")

rates["year"] = rates["TID"].str[:4].astype(int)
rates["INDHOLD"] = pd.to_numeric(rates["INDHOLD"], errors="coerce")

rates_annual = rates.pivot_table(
    index="year",
    columns="SEKTOR",
    values="INDHOLD",
    aggfunc="mean"
).reset_index()

rates_annual.columns.name = None
rates_annual = rates_annual.rename(columns={
    "year": "TID",
    "S.11 Non-financial corporations": "corporate_lending_rate_pct",
    "S.14 Households": "household_lending_rate_pct"
})

macro = macro.merge(rates_annual, on="TID", how="left")
macro.to_csv(OUT / "denmark_macro_annual.csv", index=False)

print("Saved:", OUT / "denmark_macro_annual.csv")
print("Shape:", macro.shape)
print(macro.tail(10).to_string(index=False))
