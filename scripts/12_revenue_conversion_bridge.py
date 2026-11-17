
import numpy as np
import pandas as pd
from pathlib import Path

baseline_revenue = 60.0
target_revenue = 72.0
opportunity_pool = 120.0
annual_budget = 0.135
contribution_margin = 0.25

baseline_rates = {
    "specification_capture": 0.80,
    "specification_retention": 0.80,
    "purchase_conversion": 0.78125
}

scenarios = {
    "Conservative": 0.02,
    "Base": 0.05,
    "Upside": 0.07
}

output_dir = Path("data/processed/financial_model")
output_dir.mkdir(parents=True, exist_ok=True)

years = np.arange(2026, 2031)
bridge_results = []
summary_results = []

baseline_product = np.prod(list(baseline_rates.values()))

for scenario, improvement in scenarios.items():
    previous_rates = baseline_rates.copy()

    for year in years:
        progress = (year - 2026) / 4

        rates = {
            key: min(value + improvement * progress, 1.0)
            for key, value in baseline_rates.items()
        }

        revenue_1 = opportunity_pool * rates["specification_capture"] * baseline_rates["specification_retention"] * baseline_rates["purchase_conversion"]

        revenue_2 = opportunity_pool * rates["specification_capture"] * rates["specification_retention"] * baseline_rates["purchase_conversion"]

        revenue_3 = opportunity_pool * np.prod(list(rates.values()))

        baseline_stage_1 = opportunity_pool * baseline_rates["specification_capture"] * baseline_rates["specification_retention"] * baseline_rates["purchase_conversion"]

        specification_gain = revenue_1 - baseline_stage_1
        retention_gain = revenue_2 - revenue_1
        purchase_gain = revenue_3 - revenue_2

        incremental_revenue = revenue_3 - baseline_revenue

        bridge_results.append({
            "Scenario": scenario,
            "Year": year,
            "Opportunity_Pool_EUR_m": opportunity_pool,
            "Specification_Capture": rates["specification_capture"],
            "Specification_Retention": rates["specification_retention"],
            "Purchase_Conversion": rates["purchase_conversion"],
            "Specification_Gain_EUR_m": specification_gain,
            "Retention_Gain_EUR_m": retention_gain,
            "Purchase_Conversion_Gain_EUR_m": purchase_gain,
            "Incremental_Revenue_EUR_m": incremental_revenue,
            "Total_Revenue_EUR_m": revenue_3,
            "Incremental_Contribution_EUR_m": incremental_revenue * contribution_margin
        })

        previous_rates = rates

    final_revenue = opportunity_pool * np.prod([
        value + improvement for value in baseline_rates.values()
    ])

    summary_results.append({
        "Scenario": scenario,
        "Improvement_Per_Stage_pp": improvement * 100,
        "2030_Revenue_EUR_m": final_revenue,
        "2030_Incremental_Revenue_EUR_m": final_revenue - baseline_revenue,
        "2030_Target_Gap_EUR_m": final_revenue - target_revenue,
        "2030_Contribution_EUR_m": (final_revenue - baseline_revenue) * contribution_margin,
        "Total_Program_Budget_EUR_m": annual_budget * 4,
        "Target_Achieved": final_revenue >= target_revenue
    })

bridge = pd.DataFrame(bridge_results)
summary = pd.DataFrame(summary_results)

bridge.to_csv(output_dir / "revenue_bridge.csv", index=False)
summary.to_csv(output_dir / "revenue_scenario_summary.csv", index=False)

assumptions = pd.DataFrame([
    ["2026 baseline revenue", baseline_revenue, "EUR million", "Case-provided"],
    ["2030 revenue target", target_revenue, "EUR million", "Case-provided"],
    ["Annual initiative budget", annual_budget, "EUR million", "Case-provided"],
    ["Contribution margin", contribution_margin, "Ratio", "Case assumption"],
    ["Normalized opportunity pool", opportunity_pool, "EUR million", "Calibrated assumption"],
    ["Specification capture", 0.80, "Ratio", "Hypothetical"],
    ["Specification retention", 0.80, "Ratio", "Hypothetical"],
    ["Purchase conversion", 0.78125, "Ratio", "Hypothetical"]
], columns=["Parameter", "Value", "Unit", "Classification"])

assumptions.to_csv(output_dir / "revenue_model_assumptions.csv", index=False)

print("\n===== REVENUE SCENARIOS =====")
print(summary.round(3).to_string(index=False))

print("\n===== BASE CASE ANNUAL BRIDGE =====")
print(
    bridge[bridge["Scenario"] == "Base"][
        ["Year", "Total_Revenue_EUR_m", "Specification_Gain_EUR_m",
         "Retention_Gain_EUR_m", "Purchase_Conversion_Gain_EUR_m"]
    ].round(3).to_string(index=False)
)
