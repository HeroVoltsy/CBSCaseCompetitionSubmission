
import numpy as np
import numpy_financial as npf
import pandas as pd
from pathlib import Path

bridge = pd.read_csv("data/processed/financial_model/revenue_bridge.csv")

budget = 0.135
margin = 0.25
discount_rates = [0.08, 0.10, 0.12, 0.15]
budget_multipliers = [1.0, 1.25, 1.50]
margin_scenarios = [0.15, 0.20, 0.25, 0.30]
revenue_realization = [0.50, 0.75, 1.00]

output_dir = Path("data/processed/financial_model")
output_dir.mkdir(parents=True, exist_ok=True)

required_columns = {
    "Scenario",
    "Year",
    "Incremental_Revenue_EUR_m",
    "Total_Revenue_EUR_m"
}

if not required_columns.issubset(bridge.columns):
    raise ValueError(f"Missing columns: {required_columns - set(bridge.columns)}")

if bridge.duplicated(["Scenario", "Year"]).any():
    raise ValueError("Duplicate scenario-year observations detected.")

valuation_results = []
sensitivity_results = []

for scenario in bridge["Scenario"].unique():
    data = bridge[bridge["Scenario"] == scenario].sort_values("Year")

    annual_data = data[data["Year"].between(2027, 2030)]

    if len(annual_data) != 4 or annual_data["Year"].tolist() != [2027, 2028, 2029, 2030]:
        raise ValueError(f"Incomplete annual data for {scenario}.")

    uplift = annual_data["Incremental_Revenue_EUR_m"].to_numpy(dtype=float)
    contribution = uplift * margin

    cash_flows = np.array([
        -budget,
        contribution[0] - budget,
        contribution[1] - budget,
        contribution[2] - budget,
        contribution[3]
    ])

    if not np.isfinite(cash_flows).all():
        raise ValueError(f"Invalid cash flows for {scenario}.")

    for rate in discount_rates:
        periods = np.arange(len(cash_flows))
        discounted = cash_flows / (1 + rate) ** periods

        npv = discounted.sum()
        irr = npf.irr(cash_flows)

        pv_inflows = sum(
            contribution[t] / (1 + rate) ** (t + 1)
            for t in range(4)
        )

        pv_outflows = sum(
            budget / (1 + rate) ** t
            for t in range(4)
        )

        profitability_index = pv_inflows / pv_outflows

        cumulative = np.cumsum(cash_flows)
        discounted_cumulative = np.cumsum(discounted)

        payback = next(
            (float(i) for i, value in enumerate(cumulative) if value >= 0),
            np.nan
        )

        discounted_payback = next(
            (float(i) for i, value in enumerate(discounted_cumulative) if value >= 0),
            np.nan
        )

        valuation_results.append({
            "Scenario": scenario,
            "Discount_Rate": rate,
            "NPV_EUR_m": npv,
            "IRR_pct": irr * 100 if np.isfinite(irr) else np.nan,
            "Profitability_Index": profitability_index,
            "Simple_Payback_Years": payback,
            "Discounted_Payback_Years": discounted_payback,
            "Total_Budget_EUR_m": budget * 4,
            "2030_Revenue_EUR_m": annual_data.iloc[-1]["Total_Revenue_EUR_m"]
        })

    for realization in revenue_realization:
        for margin_case in margin_scenarios:
            for budget_multiplier in budget_multipliers:
                adjusted_budget = budget * budget_multiplier
                adjusted_contribution = uplift * realization * margin_case

                adjusted_cash_flows = np.array([
                    -adjusted_budget,
                    adjusted_contribution[0] - adjusted_budget,
                    adjusted_contribution[1] - adjusted_budget,
                    adjusted_contribution[2] - adjusted_budget,
                    adjusted_contribution[3]
                ])

                for rate in discount_rates:
                    npv = npf.npv(rate, adjusted_cash_flows)

                    sensitivity_results.append({
                        "Scenario": scenario,
                        "Revenue_Realization": realization,
                        "Contribution_Margin": margin_case,
                        "Budget_Multiplier": budget_multiplier,
                        "Discount_Rate": rate,
                        "NPV_EUR_m": npv
                    })

valuation = pd.DataFrame(valuation_results)
sensitivity = pd.DataFrame(sensitivity_results)

valuation.to_csv(output_dir / "integrated_valuation.csv", index=False)
sensitivity.to_csv(output_dir / "valuation_sensitivity.csv", index=False)

base = bridge[bridge["Scenario"] == "Base"].sort_values("Year")
base_annual = base[base["Year"].between(2027, 2030)]

if len(base_annual) != 4 or base_annual["Year"].tolist() != [2027, 2028, 2029, 2030]:
    raise ValueError("Incomplete annual data for Base scenario.")

base_uplift = base_annual["Incremental_Revenue_EUR_m"].to_numpy(dtype=float)
base_contribution = base_uplift * margin

base_cash_flows = np.array([
    -budget,
    base_contribution[0] - budget,
    base_contribution[1] - budget,
    base_contribution[2] - budget,
    base_contribution[3]
])

rate = 0.10

base_npv = npf.npv(rate, base_cash_flows)

pv_budget = sum(
    budget / (1 + rate) ** t
    for t in range(4)
)

pv_contribution = sum(
    base_contribution[t] / (1 + rate) ** (t + 1)
    for t in range(4)
)

break_even_margin = pv_budget / sum(
    base_uplift[t] / (1 + rate) ** (t + 1)
    for t in range(4)
)

break_even_realization = pv_budget / pv_contribution

break_even_uplift = pv_budget / (
    margin * sum(
        1 / (1 + rate) ** (t + 1)
        for t in range(4)
    )
)

break_even_results = pd.DataFrame([{
    "Discount_Rate": rate,
    "Base_NPV_EUR_m": base_npv,
    "PV_Investment_EUR_m": pv_budget,
    "PV_Contribution_EUR_m": pv_contribution,
    "Break_Even_Margin": break_even_margin,
    "Break_Even_Revenue_Realization": break_even_realization,
    "Break_Even_Annual_Uplift_EUR_m": break_even_uplift
}])

break_even_results.to_csv(
    output_dir / "break_even_analysis.csv",
    index=False
)

print("\n===== INTEGRATED VALUATION =====")
print(valuation.round(3).to_string(index=False))

print("\n===== BASE CASE BREAK-EVEN =====")
print(f"NPV at 10%: EUR {base_npv:.3f}m")
print(f"PV of program investment: EUR {pv_budget:.3f}m")
print(f"PV of contribution: EUR {pv_contribution:.3f}m")
print(f"Break-even contribution margin: {break_even_margin:.2%}")
print(f"Break-even revenue realization: {break_even_realization:.2%}")
print(f"Break-even annual uplift: EUR {break_even_uplift:.3f}m")

print("\n===== BASE CASE SENSITIVITY AT 10% =====")
print(
    sensitivity[
        (sensitivity["Scenario"] == "Base")
        & (sensitivity["Discount_Rate"] == 0.10)
    ].round(3).to_string(index=False)
)

print("\n===== OUTPUT FILES =====")
for filename in [
    "integrated_valuation.csv",
    "valuation_sensitivity.csv",
    "break_even_analysis.csv"
]:
    print(output_dir / filename)
