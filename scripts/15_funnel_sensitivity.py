
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

input_dir = Path("data/processed/financial_model")
output_dir = input_dir / "funnel_sensitivity"
output_dir.mkdir(parents=True, exist_ok=True)

opportunity_pool = 120.0
baseline_revenue = 60.0
baseline_rates = {
    "Specification_Capture": 0.80,
    "Specification_Retention": 0.80,
    "Purchase_Conversion": 0.78125
}

base_improvements = {
    "Specification_Capture": 0.05,
    "Specification_Retention": 0.05,
    "Purchase_Conversion": 0.05
}

margin = 0.25
annual_budget = 0.135
discount_rate = 0.10
years = np.arange(2026, 2031)
levels = [0.00, 0.025, 0.05, 0.075, 0.10]


def evaluate(improvements):
    annual_revenue = []

    for year in years:
        progress = (year - 2026) / 4

        rates = {
            key: baseline_rates[key] + improvements[key] * progress
            for key in baseline_rates
        }

        if any(value > 1 or value < 0 for value in rates.values()):
            raise ValueError("Conversion rate outside valid bounds.")

        revenue = opportunity_pool * np.prod(list(rates.values()))
        annual_revenue.append(revenue)

    annual_revenue = np.array(annual_revenue)

    incremental_contribution = (
        annual_revenue[1:] - baseline_revenue
    ) * margin

    cash_flows = np.array([
        -annual_budget,
        incremental_contribution[0] - annual_budget,
        incremental_contribution[1] - annual_budget,
        incremental_contribution[2] - annual_budget,
        incremental_contribution[3]
    ])

    discount_factors = (
        1 / (1 + discount_rate) ** np.arange(5)
    )

    npv = np.sum(cash_flows * discount_factors)

    return {
        "2030_Revenue_EUR_m": annual_revenue[-1],
        "Revenue_Uplift_EUR_m": annual_revenue[-1] - baseline_revenue,
        "Target_Gap_EUR_m": annual_revenue[-1] - 72.0,
        "Target_Achieved": annual_revenue[-1] >= 72.0,
        "NPV_EUR_m": npv,
        "IRR_pct": npf_irr(cash_flows) * 100,
        "Capture_Rate": rates["Specification_Capture"],
        "Retention_Rate": rates["Specification_Retention"],
        "Purchase_Conversion_Rate": rates["Purchase_Conversion"]
    }


def npf_irr(cash_flows):
    import numpy_financial as npf
    return npf.irr(cash_flows)

base_result = evaluate(base_improvements)

if not np.isclose(base_result["2030_Revenue_EUR_m"], 72.069, atol=0.002):
    raise ValueError("Base revenue does not match validated model.")

if not np.isclose(base_result["NPV_EUR_m"], 5.127603, atol=0.002):
    raise ValueError("Base NPV does not match validated valuation.")

# One-way sensitivity
one_way_results = []

for lever in base_improvements:
    for improvement in levels:
        assumptions = base_improvements.copy()
        assumptions[lever] = improvement
        result = evaluate(assumptions)
        one_way_results.append({
            "Lever": lever,
            "Improvement_pp": improvement * 100,
            **result
        })

one_way = pd.DataFrame(one_way_results)
one_way.to_csv(output_dir / "one_way_sensitivity.csv", index=False)

# Full factorial sensitivity
factorial_results = []

for values in itertools.product(levels, repeat=3):
    assumptions = dict(zip(base_improvements.keys(), values))
    result = evaluate(assumptions)
    factorial_results.append({
        "Capture_Improvement_pp": values[0] * 100,
        "Retention_Improvement_pp": values[1] * 100,
        "Purchase_Improvement_pp": values[2] * 100,
        **result
    })

factorial = pd.DataFrame(factorial_results)
factorial.to_csv(output_dir / "factorial_sensitivity.csv", index=False)

# Lever impact relative to base case
lever_summary = (
    one_way.groupby("Lever")
    .agg(
        Minimum_NPV=("NPV_EUR_m", "min"),
        Maximum_NPV=("NPV_EUR_m", "max"),
        Minimum_Revenue=("2030_Revenue_EUR_m", "min"),
        Maximum_Revenue=("2030_Revenue_EUR_m", "max")
    )
    .reset_index()
)

lever_summary["NPV_Range"] = (
    lever_summary["Maximum_NPV"] - lever_summary["Minimum_NPV"]
)

lever_summary = lever_summary.sort_values(
    "NPV_Range", ascending=False
)

lever_summary.to_csv(
    output_dir / "lever_impact_summary.csv", index=False
)

# Best and worst combinations
best_cases = factorial.nlargest(10, "NPV_EUR_m")
worst_cases = factorial.nsmallest(10, "NPV_EUR_m")

best_cases.to_csv(output_dir / "best_combinations.csv", index=False)
worst_cases.to_csv(output_dir / "worst_combinations.csv", index=False)

# Revenue and NPV response by lever
for lever in base_improvements:
    data = one_way[one_way["Lever"] == lever]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        data["Improvement_pp"],
        data["2030_Revenue_EUR_m"],
        marker="o",
        label="2030 Revenue"
    )
    ax1.axhline(72, linestyle="--", label="Revenue Target")
    ax1.set_xlabel("Improvement over baseline (percentage points)")
    ax1.set_ylabel("2030 Revenue (EUR millions)")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        data["Improvement_pp"],
        data["NPV_EUR_m"],
        marker="s",
        linestyle=":",
        label="NPV"
    )
    ax2.set_ylabel("NPV (EUR millions)")

    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.title(f"Financial Sensitivity: {lever.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{lever.lower()}_sensitivity.png",
        dpi=300
    )
    plt.close()

print("\n===== BASE CASE VALIDATION =====")
print(pd.Series(base_result).to_string())

print("\n===== LEVER IMPACT SUMMARY =====")
print(lever_summary.to_string(index=False))

print("\n===== FACTORIAL SENSITIVITY =====")
print(f"Combinations evaluated: {len(factorial)}")
print(f"Minimum NPV: €{factorial['NPV_EUR_m'].min():.3f}m")
print(f"Maximum NPV: €{factorial['NPV_EUR_m'].max():.3f}m")
print(f"Target-achieving combinations: {factorial['Target_Achieved'].sum()}")
print(f"Target attainment rate: {factorial['Target_Achieved'].mean():.1%}")

print("\n===== BEST COMBINATIONS =====")
print(best_cases[
    [
        "Capture_Improvement_pp",
        "Retention_Improvement_pp",
        "Purchase_Improvement_pp",
        "2030_Revenue_EUR_m",
        "NPV_EUR_m"
    ]
].to_string(index=False))

print("\n===== OUTPUT FILES =====")
for path in sorted(output_dir.iterdir()):
    print(path)
