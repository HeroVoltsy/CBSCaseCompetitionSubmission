
from pathlib import Path
import pandas as pd

path = Path("data/processed/financials/rockwool_xbrl_facts.csv")
df = pd.read_csv(path)

patterns = [
    "Revenue",
    "Sales",
    "OperatingActivities",
    "OperatingProfit",
    "ProfitLossFromOperating",
    "EBITDA",
    "Depreciation",
    "CapitalExpenditure",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "Assets",
    "Borrowings",
    "LeaseLiabilities",
    "CashAndCashEquivalents",
    "Equity",
    "FinanceCosts"
]

mask = df["concept"].fillna("").str.contains(
    "|".join(patterns),
    case=False,
    regex=True
)

result = df.loc[mask].copy()

result = result[
    [
        "concept",
        "numeric_value",
        "unit",
        "period_start",
        "period_end",
        "period_instant",
        "dimensions",
        "decimals"
    ]
].sort_values(["concept", "period_end", "period_instant"])

result.to_csv(
    "data/processed/financials/rockwool_financial_candidates.csv",
    index=False
)

pd.set_option("display.max_rows", 250)
pd.set_option("display.max_colwidth", 100)

print(result.to_string(index=False))
print("\nSaved candidate facts.")
