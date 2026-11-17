
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

input_dir = Path("data/processed/financial_model")
output_dir = input_dir / "reconciliation"
output_dir.mkdir(parents=True, exist_ok=True)

bridge = pd.read_csv(input_dir / "revenue_bridge.csv")
sensitivity = pd.read_csv(input_dir / "valuation_sensitivity.csv")
valuation = pd.read_csv(input_dir / "integrated_valuation.csv")

budget = 0.135
margin = 0.25
discount_rate = 0.10
baseline_revenue = 60.0

annual_results = []
cashflow_results = []

for scenario in bridge["Scenario"].unique():
    data = bridge[bridge["Scenario"] == scenario].sort_values("Year")

    annual = data[data["Year"].between(2027, 2030)]

    if annual["Year"].tolist() != [2027, 2028, 2029, 2030]:
        raise ValueError(f"Incomplete annual revenue data: {scenario}")

    uplift = annual["Incremental_Revenue_EUR_m"].to_numpy(dtype=float)
    revenue = annual["Total_Revenue_EUR_m"].to_numpy(dtype=float)
    contribution = uplift * margin

    for i, year in enumerate(range(2027, 2031)):
        annual_results.append({
            "Scenario": scenario,
            "Year": year,
            "Total_Revenue_EUR_m": revenue[i],
            "Incremental_Revenue_EUR_m": uplift[i],
            "Contribution_Margin": margin,
            "Incremental_Contribution_EUR_m": contribution[i],
            "Annual_Program_Budget_EUR_m": budget,
            "Annual_Net_Contribution_EUR_m": contribution[i] - budget
        })

    cash_flows = np.array([
        -budget,
        contribution[0] - budget,
        contribution[1] - budget,
        contribution[2] - budget,
        contribution[3]
    ])

    for period in range(5):
        revenue_year = 2026 + period
        budget_year = 2027 + period if period < 4 else None

        contribution_value = contribution[period - 1] if period > 0 else 0.0
        investment_value = budget if period < 4 else 0.0

        cashflow_results.append({
            "Scenario": scenario,
            "Period": period,
            "Revenue_Year": revenue_year if period > 0 else None,
            "Budget_Year": budget_year,
            "Incremental_Contribution_EUR_m": contribution_value,
            "Program_Investment_EUR_m": investment_value,
            "Net_Cash_Flow_EUR_m": cash_flows[period],
            "Discount_Factor": 1 / (1 + discount_rate) ** period,
            "Discounted_Net_Cash_Flow_EUR_m":
                cash_flows[period] / (1 + discount_rate) ** period
        })

annual_reconciliation = pd.DataFrame(annual_results)
cashflow_reconciliation = pd.DataFrame(cashflow_results)

cashflow_reconciliation["Cumulative_Net_Cash_Flow_EUR_m"] = (
    cashflow_reconciliation.groupby("Scenario")["Net_Cash_Flow_EUR_m"].cumsum()
)

cashflow_reconciliation["Cumulative_Discounted_CF_EUR_m"] = (
    cashflow_reconciliation.groupby("Scenario")[
        "Discounted_Net_Cash_Flow_EUR_m"
    ].cumsum()
)

annual_reconciliation.to_csv(
    output_dir / "annual_revenue_reconciliation.csv",
    index=False
)

cashflow_reconciliation.to_csv(
    output_dir / "cashflow_reconciliation.csv",
    index=False
)

npv_reconciliation = (
    cashflow_reconciliation.groupby("Scenario")[
        "Discounted_Net_Cash_Flow_EUR_m"
    ].sum().reset_index(name="Recalculated_NPV_EUR_m")
)

reference_npv = valuation[valuation["Discount_Rate"] == discount_rate][
    ["Scenario", "NPV_EUR_m"]
]

npv_reconciliation = npv_reconciliation.merge(
    reference_npv,
    on="Scenario",
    how="left"
)

npv_reconciliation["NPV_Difference_EUR_m"] = (
    npv_reconciliation["Recalculated_NPV_EUR_m"]
    - npv_reconciliation["NPV_EUR_m"]
)

if not np.allclose(
    npv_reconciliation["NPV_Difference_EUR_m"],
    0,
    atol=0.001
):
    raise ValueError("Cash-flow reconciliation does not match integrated valuation.")

npv_reconciliation.to_csv(
    output_dir / "npv_reconciliation.csv",
    index=False
)

base_sensitivity = sensitivity[
    (sensitivity["Scenario"] == "Base")
    & (sensitivity["Discount_Rate"] == discount_rate)
    & (sensitivity["Budget_Multiplier"] == 1.0)
]

heatmap_data = base_sensitivity.pivot(
    index="Revenue_Realization",
    columns="Contribution_Margin",
    values="NPV_EUR_m"
)

heatmap_data.to_csv(output_dir / "base_npv_sensitivity_matrix.csv")

plt.figure(figsize=(9, 5))

for scenario in bridge["Scenario"].unique():
    data = bridge[bridge["Scenario"] == scenario].sort_values("Year")

    plt.plot(
        data["Year"],
        data["Total_Revenue_EUR_m"],
        marker="o",
        label=scenario
    )

plt.axhline(72, linestyle="--", label="2030 Revenue Target")
plt.ylabel("Revenue (EUR millions)")
plt.xlabel("Year")
plt.title("Danish GBI Revenue Trajectory")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(output_dir / "revenue_trajectory.png", dpi=300)
plt.close()

base_cashflow = cashflow_reconciliation[
    cashflow_reconciliation["Scenario"] == "Base"
]

plt.figure(figsize=(9, 5))

plt.bar(
    base_cashflow["Period"],
    base_cashflow["Net_Cash_Flow_EUR_m"]
)

plt.axhline(0, linewidth=0.8)
plt.xticks(range(5), ["Initial", "2027", "2028", "2029", "2030"])
plt.ylabel("Net cash flow (EUR millions)")
plt.xlabel("Cash-flow period")
plt.title("Base-Case Program Cash Flows")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(output_dir / "base_cashflow.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))

image = plt.imshow(
    heatmap_data.to_numpy(),
    aspect="auto",
    origin="lower"
)

plt.colorbar(image, label="NPV (EUR millions)")
plt.xticks(
    range(len(heatmap_data.columns)),
    [f"{x:.0%}" for x in heatmap_data.columns]
)
plt.yticks(
    range(len(heatmap_data.index)),
    [f"{x:.0%}" for x in heatmap_data.index]
)
plt.xlabel("Contribution Margin")
plt.ylabel("Revenue Realization")
plt.title("Base-Case NPV Sensitivity at 10% Discount Rate")
plt.tight_layout()
plt.savefig(output_dir / "base_npv_sensitivity.png", dpi=300)
plt.close()

print("\n===== ANNUAL REVENUE RECONCILIATION =====")
print(annual_reconciliation.round(3).to_string(index=False))

print("\n===== BASE-CASE CASH FLOW RECONCILIATION =====")
print(
    cashflow_reconciliation[
        cashflow_reconciliation["Scenario"] == "Base"
    ].round(3).to_string(index=False)
)

print("\n===== NPV RECONCILIATION =====")
print(npv_reconciliation.round(6).to_string(index=False))

print("\n===== BASE-CASE NPV SENSITIVITY MATRIX =====")
print(heatmap_data.round(3).to_string())

print("\n===== OUTPUT FILES =====")
for file in sorted(output_dir.iterdir()):
    print(file)
