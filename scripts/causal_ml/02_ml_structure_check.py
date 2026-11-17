
import pandas as pd
from pathlib import Path

base = Path("data/processed/market_indicators")

activity = pd.read_csv(base / "construction_activity_annual.csv")
quarterly = pd.read_csv(base / "construction_activity_quarterly.csv")
by_use = pd.read_csv(base / "construction_by_use.csv")
stock_age = pd.read_csv(base / "building_stock_by_age.csv")
stock_use = pd.read_csv(base / "building_stock_by_use.csv")
renovation = pd.read_csv(base / "renovation_cost_index.csv")

print("\nANNUAL ACTIVITY")
print("Years:", activity["TID"].min(), "-", activity["TID"].max())
print(activity.tail(10).to_string(index=False))

print("\nQUARTERLY ACTIVITY")
print("Periods:", quarterly["TID"].min(), "-", quarterly["TID"].max())
print("Number of periods:", quarterly["TID"].nunique())
print(quarterly.groupby("BYGFASE")["TID"].nunique())

print("\nBUILDING USE PANEL")
print("Years:", by_use["TID"].min(), "-", by_use["TID"].max())
print("Building-use categories:", by_use["ANVEND"].nunique())
print("Observations per phase:")
print(by_use.groupby("BYGFASE").size())
print("\nCategories:")
print(sorted(by_use["ANVEND"].unique()))

print("\nSTARTED ACTIVITY BY USE")
started = by_use[by_use["BYGFASE"] == "Started"]
print(
    started.groupby("ANVEND")["INDHOLD"]
    .agg(["mean", "median", "min", "max", "sum"])
    .sort_values("sum", ascending=False)
    .round(2)
    .to_string()
)

print("\nBUILDING STOCK")
print("Age categories:", stock_age["OPFØRELSESÅR"].nunique())
print("Use categories:", stock_use["ANVEND"].nunique())
print("Age stock years:", stock_age["TID"].min(), "-", stock_age["TID"].max())
print("Use stock years:", stock_use["TID"].min(), "-", stock_use["TID"].max())

print("\nRENOVATION COST INDEX")
print(
    renovation.pivot(
        index="TID",
        columns="TAL",
        values="INDHOLD"
    ).round(3).to_string()
)

print("\nMISSING YEARS IN ANNUAL ACTIVITY")
expected = set(range(activity["TID"].min(), activity["TID"].max() + 1))
print(sorted(expected - set(activity["TID"])))
