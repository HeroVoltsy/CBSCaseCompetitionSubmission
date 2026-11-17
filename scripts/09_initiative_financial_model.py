
import numpy as np
import numpy_financial as npf
import pandas as pd

baseline = 60.0
target = 72.0
margin = 0.25
annual_budget = 0.135
discount_rates = [0.08, 0.10, 0.12, 0.15]

scenarios = {
    "Conservative": [1, 3, 6, 9],
    "Base": [3, 6, 9, 12],
    "Upside": [4, 8, 11, 12]
}

results = []

for scenario, revenue_uplift in scenarios.items():
    contribution = np.array(revenue_uplift) * margin

    cash_flows = np.array([
        -annual_budget,
        contribution[0] - annual_budget,
        contribution[1] - annual_budget,
        contribution[2] - annual_budget,
        contribution[3]
    ])

    for rate in discount_rates:
        npv = npf.npv(rate, cash_flows)
        irr = npf.irr(cash_flows)

        results.append({
            "Scenario": scenario,
            "Discount Rate": rate,
            "NPV_EUR_m": npv,
            "IRR_pct": irr * 100,
            "Total_Budget_EUR_m": annual_budget * 4,
            "2030_Revenue_EUR_m": baseline + revenue_uplift[-1],
            "2030_Incremental_Contribution_EUR_m": contribution[-1]
        })

results = pd.DataFrame(results)
results.to_csv(
    "data/processed/initiative_valuation.csv",
    index=False
)

print(results.round(3).to_string(index=False))
