
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data/processed/causal_ml/"
    / "dml/strict_pre_treatment/"
    / "strict_dml_predictions.csv"
)

OUT = (
    ROOT / "data/processed/causal_ml/"
    / "dml/strict_pre_treatment/inference"
)

OUT.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 10000
BLOCK_LENGTHS = [2, 3]
SEED = 42


def estimate_theta(frame):
    residual_t = frame["Residual_Permits"].to_numpy()
    residual_y = frame["Residual_Started"].to_numpy()

    denominator = np.sum(residual_t ** 2)

    if denominator <= 0:
        return np.nan

    return np.sum(
        residual_t * residual_y
    ) / denominator


def aggregate_years(frame):
    frame = frame.copy()

    frame["Numerator"] = (
        frame["Residual_Permits"]
        * frame["Residual_Started"]
    )

    frame["Denominator"] = (
        frame["Residual_Permits"] ** 2
    )

    return (
        frame.groupby("Year")
        .agg(
            Numerator=("Numerator", "sum"),
            Denominator=("Denominator", "sum"),
        )
        .sort_index()
    )


def circular_block_bootstrap(
    yearly,
    block_length,
    n_bootstrap,
    rng,
):
    numerator = yearly["Numerator"].to_numpy()
    denominator = yearly["Denominator"].to_numpy()

    n_years = len(yearly)
    estimates = []

    for _ in range(n_bootstrap):
        indices = []

        while len(indices) < n_years:
            start = rng.integers(0, n_years)

            block = [
                (start + offset) % n_years
                for offset in range(block_length)
            ]

            indices.extend(block)

        indices = np.asarray(indices[:n_years])

        sampled_numerator = numerator[indices].sum()
        sampled_denominator = denominator[indices].sum()

        if sampled_denominator > 0:
            estimates.append(
                sampled_numerator / sampled_denominator
            )

    return np.asarray(estimates)


def leave_one_year_out(frame, model_name, full_estimate):
    results = []

    for year in sorted(frame["Year"].unique()):
        reduced = frame[
            frame["Year"] != year
        ]

        estimate = estimate_theta(reduced)

        results.append({
            "Model": model_name,
            "Removed_Year": year,
            "Estimate": estimate,
            "Change_From_Full": (
                estimate - full_estimate
            ),
        })

    return pd.DataFrame(results)


def main():
    df = pd.read_csv(INPUT)

    required = [
        "Year",
        "Category",
        "Residual_Started",
        "Residual_Permits",
        "Model",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[required].dropna().copy()

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="raise",
    ).astype(int)

    rng = np.random.default_rng(SEED)

    all_loo = []
    all_bootstrap = []
    all_yearly = []

    for model_name, model_df in df.groupby("Model"):
        model_df = model_df.copy()

        full_estimate = estimate_theta(model_df)

        yearly = aggregate_years(model_df)

        print(f"\n===== {model_name} =====")
        print("Full estimate:", full_estimate)
        print("Year clusters:", len(yearly))

        # Leave-one-year-out analysis
        loo = leave_one_year_out(
            model_df,
            model_name,
            full_estimate,
        )

        all_loo.append(loo)

        print(
            "LOO range:",
            round(loo["Estimate"].min(), 6),
            round(loo["Estimate"].max(), 6),
        )

        print(
            "Maximum absolute influence:",
            round(
                loo["Change_From_Full"].abs().max(),
                6,
            ),
        )

        # Save year-level contributions
        yearly_output = yearly.reset_index()
        yearly_output["Model"] = model_name

        all_yearly.append(yearly_output)

        # Corrected circular moving-block bootstrap
        for block_length in BLOCK_LENGTHS:
            estimates = circular_block_bootstrap(
                yearly,
                block_length,
                N_BOOTSTRAP,
                rng,
            )

            lower, upper = np.quantile(
                estimates,
                [0.025, 0.975],
            )

            result = {
                "Model": model_name,
                "Block_Length": block_length,
                "N_Bootstrap": len(estimates),
                "Original_Estimate": full_estimate,
                "Bootstrap_Mean": estimates.mean(),
                "Bootstrap_SD": estimates.std(ddof=1),
                "CI_Lower_95": lower,
                "CI_Upper_95": upper,
                "Positive_Estimate_Share": np.mean(
                    estimates > 0
                ),
            }

            all_bootstrap.append(result)

            print(
                f"Block {block_length}: "
                f"Mean={estimates.mean():.5f} | "
                f"95% CI=[{lower:.5f}, {upper:.5f}] | "
                f"Positive share="
                f"{result['Positive_Estimate_Share']:.3f}"
            )

    # Combine outputs
    loo_all = pd.concat(
        all_loo,
        ignore_index=True,
    )

    bootstrap_all = pd.DataFrame(
        all_bootstrap
    )

    yearly_all = pd.concat(
        all_yearly,
        ignore_index=True,
    )

    # Save
    loo_all.to_csv(
        OUT / "leave_one_year_out_all.csv",
        index=False,
    )

    bootstrap_all.to_csv(
        OUT / "corrected_moving_block_bootstrap.csv",
        index=False,
    )

    yearly_all.to_csv(
        OUT / "year_level_contributions.csv",
        index=False,
    )

    # Summary
    print("\n===== LEAVE-ONE-YEAR-OUT SUMMARY =====")

    print(
        loo_all.groupby("Model")["Estimate"]
        .agg(["min", "max", "mean"])
        .to_string()
    )

    print("\n===== CORRECTED BOOTSTRAP SUMMARY =====")

    print(
        bootstrap_all.to_string(index=False)
    )

    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()
