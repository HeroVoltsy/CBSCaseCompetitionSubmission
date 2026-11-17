
from pathlib import Path

import warnings
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "causal_ml"
CPCV_DIR = DATA_DIR / "cpcv"
OUT_DIR = CPCV_DIR / "inference"

OUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = CPCV_DIR / "cpcv_predictions.csv"

RANDOM_STATE = 42
BOOTSTRAP_REPS = 3000
BLOCK_LENGTH = 2
CONFIDENCE = 0.95

BASELINE = "Persistence"

MODELS = [
    "Ridge",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def load_predictions():
    df = pd.read_csv(PREDICTIONS_FILE)

    required = [
        "Dataset",
        "Split",
        "TID",
        "ANVEND",
        "Model",
        "Started",
        "Prediction",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    df["TID"] = df["TID"].astype(int)
    df["Started"] = pd.to_numeric(df["Started"], errors="coerce")
    df["Prediction"] = pd.to_numeric(
        df["Prediction"], errors="coerce"
    )

    df = df.dropna(
        subset=["Started", "Prediction"]
    )

    # CPCV observations recur across test splits.
    # Average predictions before statistical inference.
    unique = (
        df.groupby(
            ["Dataset", "Model", "TID", "ANVEND"],
            as_index=False,
        )
        .agg(
            Started=("Started", "mean"),
            Prediction=("Prediction", "mean"),
            Prediction_SD=("Prediction", "std"),
            Split_Count=("Split", "nunique"),
        )
    )

    unique["Residual"] = (
        unique["Started"] - unique["Prediction"]
    )

    unique["Absolute_Error"] = unique["Residual"].abs()
    unique["Squared_Error"] = unique["Residual"] ** 2

    return unique


def annual_metrics(df):
    records = []

    for keys, group in df.groupby(
        ["Dataset", "Model", "TID"]
    ):
        dataset, model, year = keys

        y = group["Started"].to_numpy()
        pred = group["Prediction"].to_numpy()

        error = y - pred

        denominator = np.sum(np.abs(y))

        records.append({
            "Dataset": dataset,
            "Model": model,
            "Year": year,
            "N": len(group),
            "MAE": np.mean(np.abs(error)),
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "WAPE_pct": (
                np.sum(np.abs(error)) / denominator * 100
                if denominator > 0 else np.nan
            ),
            "Bias": np.mean(pred - y),
        })

    return pd.DataFrame(records)


def cohen_dz(differences):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    if len(differences) < 2:
        return np.nan

    sd = np.std(differences, ddof=1)

    if sd == 0:
        return np.nan

    return np.mean(differences) / sd


def circular_block_bootstrap(values, rng, reps, block_length):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n == 0:
        return np.array([])

    block_length = min(block_length, n)
    blocks_needed = int(np.ceil(n / block_length))

    estimates = []

    for _ in range(reps):
        indices = []

        for _ in range(blocks_needed):
            start = rng.integers(0, n)
            block = [
                (start + j) % n
                for j in range(block_length)
            ]
            indices.extend(block)

        indices = indices[:n]
        sample = values[indices]

        estimates.append({
            "mean": np.mean(sample),
            "d_z": cohen_dz(sample),
        })

    return pd.DataFrame(estimates)


def bootstrap_effect(differences, rng):
    boot = circular_block_bootstrap(
        differences,
        rng,
        BOOTSTRAP_REPS,
        BLOCK_LENGTH,
    )

    alpha = 1 - CONFIDENCE

    result = {
        "Mean_Improvement": np.mean(differences),
        "Cohen_dz": cohen_dz(differences),
        "Bootstrap_Mean_Lower": boot["mean"].quantile(alpha / 2),
        "Bootstrap_Mean_Upper": boot["mean"].quantile(1 - alpha / 2),
        "Bootstrap_dz_Lower": boot["d_z"].quantile(alpha / 2),
        "Bootstrap_dz_Upper": boot["d_z"].quantile(1 - alpha / 2),
    }

    return result


def safe_wilcoxon(a, b):
    difference = np.asarray(a) - np.asarray(b)

    difference = difference[np.isfinite(difference)]

    if len(difference) < 5 or np.allclose(difference, 0):
        return np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = wilcoxon(
            difference,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

    return result.statistic, result.pvalue


def compare_against_persistence(annual, rng):
    records = []

    for dataset, data in annual.groupby("Dataset"):
        pivot = data.pivot(
            index="Year",
            columns="Model",
            values=["MAE", "RMSE", "WAPE_pct"],
        ).sort_index()

        for model in MODELS:
            if model not in pivot["RMSE"].columns:
                continue

            for metric in ["MAE", "RMSE", "WAPE_pct"]:
                paired = pivot[metric][
                    [BASELINE, model]
                ].dropna()

                baseline_values = paired[BASELINE].to_numpy()
                model_values = paired[model].to_numpy()

                # Positive means the candidate model improves
                # on persistence.
                differences = baseline_values - model_values

                statistic, pvalue = safe_wilcoxon(
                    baseline_values,
                    model_values,
                )

                effect = bootstrap_effect(differences, rng)

                records.append({
                    "Dataset": dataset,
                    "Comparison": f"{model}_vs_{BASELINE}",
                    "Metric": metric,
                    "N_Years": len(differences),
                    "Wilcoxon_statistic": statistic,
                    "P_raw": pvalue,
                    **effect,
                    "Improvement_Share": np.mean(differences > 0),
                })

    result = pd.DataFrame(records)

    result["P_Holm"] = np.nan

    # Correct separately within each dataset and metric family.
    for _, indices in result.groupby(
        ["Dataset", "Metric"]
    ).groups.items():
        p = result.loc[indices, "P_raw"].to_numpy()
        valid = np.isfinite(p)

        if valid.any():
            adjusted = multipletests(
                p[valid],
                method="holm",
            )[1]

            result.loc[
                np.asarray(indices)[valid],
                "P_Holm",
            ] = adjusted

    return result


def compare_macro_features(annual, rng):
    records = []

    full_name = "Full_History"
    macro_name = "Macro_Enhanced"

    for model in ["Persistence"] + MODELS:
        for metric in ["MAE", "RMSE", "WAPE_pct"]:
            subset = annual[
                annual["Model"] == model
            ]

            pivot = subset.pivot(
                index="Year",
                columns="Dataset",
                values=metric,
            ).sort_index()

            if (
                full_name not in pivot.columns
                or macro_name not in pivot.columns
            ):
                continue

            paired = pivot[
                [full_name, macro_name]
            ].dropna()

            full = paired[full_name].to_numpy()
            macro = paired[macro_name].to_numpy()

            # Positive means macro features reduce error.
            differences = full - macro

            statistic, pvalue = safe_wilcoxon(full, macro)
            effect = bootstrap_effect(differences, rng)

            records.append({
                "Model": model,
                "Metric": metric,
                "N_Years": len(differences),
                "Wilcoxon_statistic": statistic,
                "P_raw": pvalue,
                **effect,
                "Improvement_Share": np.mean(differences > 0),
            })

    result = pd.DataFrame(records)

    result["P_Holm"] = np.nan

    if not result.empty:
        for metric, indices in result.groupby("Metric").groups.items():
            p = result.loc[indices, "P_raw"].to_numpy()
            valid = np.isfinite(p)

            if valid.any():
                adjusted = multipletests(
                    p[valid],
                    method="holm",
                )[1]

                result.loc[
                    np.asarray(indices)[valid],
                    "P_Holm",
                ] = adjusted

    return result


def correlation_analysis(unique):
    prediction_records = []
    residual_records = []

    for dataset, data in unique.groupby("Dataset"):
        pred = data.pivot(
            index=["TID", "ANVEND"],
            columns="Model",
            values="Prediction",
        )

        resid = data.pivot(
            index=["TID", "ANVEND"],
            columns="Model",
            values="Residual",
        )

        for model_a, model_b in combinations_safe(pred.columns):
            p = pred[[model_a, model_b]].dropna()
            r = resid[[model_a, model_b]].dropna()

            if len(p) < 5:
                continue

            prediction_records.append({
                "Dataset": dataset,
                "Model_A": model_a,
                "Model_B": model_b,
                "N": len(p),
                "Pearson_r": pearsonr(
                    p[model_a], p[model_b]
                ).statistic,
                "Spearman_rho": spearmanr(
                    p[model_a], p[model_b]
                ).statistic,
            })

            residual_records.append({
                "Dataset": dataset,
                "Model_A": model_a,
                "Model_B": model_b,
                "N": len(r),
                "Residual_Pearson_r": pearsonr(
                    r[model_a], r[model_b]
                ).statistic,
                "Residual_Spearman_rho": spearmanr(
                    r[model_a], r[model_b]
                ).statistic,
            })

    return (
        pd.DataFrame(prediction_records),
        pd.DataFrame(residual_records),
    )


def combinations_safe(columns):
    from itertools import combinations
    return combinations(list(columns), 2)


def category_analysis(unique):
    records = []

    for keys, group in unique.groupby(
        ["Dataset", "Model", "ANVEND"]
    ):
        dataset, model, category = keys

        y = group["Started"].to_numpy()
        pred = group["Prediction"].to_numpy()
        error = y - pred

        denominator = np.sum(np.abs(y))

        records.append({
            "Dataset": dataset,
            "Model": model,
            "ANVEND": category,
            "N_Years": len(group),
            "MAE": np.mean(np.abs(error)),
            "RMSE": np.sqrt(np.mean(error ** 2)),
            "WAPE_pct": (
                np.sum(np.abs(error)) / denominator * 100
                if denominator > 0 else np.nan
            ),
            "Bias": np.mean(pred - y),
        })

    result = pd.DataFrame(records)

    baseline = result[
        result["Model"] == BASELINE
    ][["Dataset", "ANVEND", "RMSE", "MAE"]].rename(
        columns={
            "RMSE": "Persistence_RMSE",
            "MAE": "Persistence_MAE",
        }
    )

    result = result.merge(
        baseline,
        on=["Dataset", "ANVEND"],
        how="left",
    )

    result["RMSE_Improvement_pct"] = (
        (result["Persistence_RMSE"] - result["RMSE"])
        / result["Persistence_RMSE"] * 100
    )

    result["MAE_Improvement_pct"] = (
        (result["Persistence_MAE"] - result["MAE"])
        / result["Persistence_MAE"] * 100
    )

    return result


def main():
    rng = np.random.default_rng(RANDOM_STATE)

    print("Loading CPCV predictions...")
    unique = load_predictions()

    print("Calculating annual metrics...")
    annual = annual_metrics(unique)

    print("Paired model inference...")
    model_tests = compare_against_persistence(
        annual, rng
    )

    print("Macro incremental-value inference...")
    macro_tests = compare_macro_features(
        annual, rng
    )

    print("Model correlations...")
    prediction_corr, residual_corr = correlation_analysis(
        unique
    )

    print("Category diagnostics...")
    categories = category_analysis(unique)

    outputs = {
        "unique_predictions.csv": unique,
        "annual_metrics.csv": annual,
        "model_vs_persistence_inference.csv": model_tests,
        "macro_incremental_inference.csv": macro_tests,
        "prediction_correlations.csv": prediction_corr,
        "residual_correlations.csv": residual_corr,
        "category_effects.csv": categories,
    }

    for filename, frame in outputs.items():
        frame.to_csv(OUT_DIR / filename, index=False)

    print("\n===== MODEL VS PERSISTENCE =====")
    print(model_tests.round(4).to_string(index=False))

    print("\n===== MACRO FEATURE INFERENCE =====")
    print(macro_tests.round(4).to_string(index=False))

    print("\n===== PREDICTION CORRELATIONS =====")
    print(prediction_corr.round(4).to_string(index=False))

    print("\n===== CATEGORY-LEVEL EFFECTS =====")
    print(
        categories.sort_values(
            ["Dataset", "ANVEND", "RMSE"]
        ).round(4).to_string(index=False)
    )

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
