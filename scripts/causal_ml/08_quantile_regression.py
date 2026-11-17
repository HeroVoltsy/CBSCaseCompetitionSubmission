
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "causal_ml"
INPUT_FILE = DATA_DIR / "construction_panel_macro_modeling.csv"

OUT_DIR = DATA_DIR / "quantile_regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
BOOTSTRAP_REPS = 300
QUANTILES = [0.10, 0.50, 0.90]
CONFIDENCE = 0.95

TARGET = "Started"

NUMERIC_FEATURES = [
    "Started_lag1",
    "Started_lag2",
    "Permits_lag1",
    "Completed_lag1",
    "Under construction_lag1",
    "gdp_growth_pct_lag1",
    "corporate_lending_rate_pct_lag1",
    "household_lending_rate_pct_lag1",
    "inflation_pct_lag1",
]


def load_data():
    df = pd.read_csv(INPUT_FILE)

    required = [
        "TID",
        "ANVEND",
        TARGET,
        *NUMERIC_FEATURES,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in ["TID", TARGET] + NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ANVEND"] = df["ANVEND"].astype(str)

    df = df.dropna(subset=required).copy()
    df["TID"] = df["TID"].astype(int)

    df = df.sort_values(
        ["TID", "ANVEND"]
    ).reset_index(drop=True)

    return df


def build_design_matrix(df, numeric_means=None, numeric_stds=None):
    numeric = df[NUMERIC_FEATURES].astype(float).copy()

    if numeric_means is None:
        numeric_means = numeric.mean()

    if numeric_stds is None:
        numeric_stds = numeric.std(ddof=0).replace(0, 1)

    numeric = (numeric - numeric_means) / numeric_stds

    categories = pd.get_dummies(
        df["ANVEND"],
        prefix="category",
        drop_first=True,
        dtype=float,
    )

    X = pd.concat(
        [
            numeric.reset_index(drop=True),
            categories.reset_index(drop=True),
        ],
        axis=1,
    )

    X = sm.add_constant(X, has_constant="add")

    y = df[TARGET].to_numpy(dtype=float) / 1000.0

    return X.astype(float), y, numeric_means, numeric_stds


def fit_quantile_models(X, y):
    fitted = {}

    for q in QUANTILES:
        model = sm.QuantReg(y, X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            result = model.fit(
                q=q,
                max_iter=5000,
                p_tol=1e-6,
            )

        fitted[q] = result

    return fitted


def bootstrap_by_year(df, rng):
    years = np.array(sorted(df["TID"].unique()))

    samples = []

    for _ in range(BOOTSTRAP_REPS):
        sampled_years = rng.choice(
            years,
            size=len(years),
            replace=True,
        )

        parts = []

        for draw_id, year in enumerate(sampled_years):
            block = df[df["TID"] == year].copy()

            # Give repeated draws unique identifiers.
            block["BOOTSTRAP_DRAW"] = draw_id

            parts.append(block)

        sample = pd.concat(
            parts,
            ignore_index=True,
        )

        samples.append(sample)

    return samples


def bootstrap_coefficients(
    df,
    numeric_means,
    numeric_stds,
    feature_names,
):
    rng = np.random.default_rng(RANDOM_STATE)

    samples = bootstrap_by_year(df, rng)

    records = []

    for bootstrap_id, sample in enumerate(samples, start=1):
        X, y, _, _ = build_design_matrix(
            sample,
            numeric_means,
            numeric_stds,
        )

        X = X.reindex(
            columns=feature_names,
            fill_value=0.0,
        )

        for q in QUANTILES:
            try:
                model = sm.QuantReg(y, X)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    result = model.fit(
                        q=q,
                        max_iter=5000,
                        p_tol=1e-6,
                    )

                for feature, coefficient in result.params.items():
                    records.append({
                        "Bootstrap": bootstrap_id,
                        "Quantile": q,
                        "Feature": feature,
                        "Coefficient": coefficient,
                    })

            except Exception:
                continue

        if bootstrap_id % 25 == 0:
            print(
                f"Completed bootstrap "
                f"{bootstrap_id}/{BOOTSTRAP_REPS}"
            )

    return pd.DataFrame(records)


def coefficient_summary(original, bootstrap):
    alpha = 1 - CONFIDENCE

    records = []

    for q in QUANTILES:
        for feature, coefficient in original[q].params.items():
            subset = bootstrap[
                (bootstrap["Quantile"] == q)
                & (bootstrap["Feature"] == feature)
            ]["Coefficient"].dropna()

            if len(subset) < 30:
                lower = np.nan
                upper = np.nan
            else:
                lower = subset.quantile(alpha / 2)
                upper = subset.quantile(1 - alpha / 2)

            records.append({
                "Quantile": q,
                "Feature": feature,
                "Coefficient": coefficient,
                "Bootstrap_Lower_95": lower,
                "Bootstrap_Upper_95": upper,
                "Successful_Bootstraps": len(subset),
                "CI_Excludes_Zero": (
                    lower > 0 or upper < 0
                    if np.isfinite(lower) and np.isfinite(upper)
                    else False
                ),
            })

    return pd.DataFrame(records)


def quantile_difference_analysis(bootstrap):
    alpha = 1 - CONFIDENCE

    wide = bootstrap.pivot_table(
        index=["Bootstrap", "Feature"],
        columns="Quantile",
        values="Coefficient",
    ).reset_index()

    records = []

    for feature, group in wide.groupby("Feature"):
        if 0.10 not in group.columns or 0.90 not in group.columns:
            continue

        differences = (
            group[0.90] - group[0.10]
        ).dropna()

        if len(differences) < 30:
            continue

        lower = differences.quantile(alpha / 2)
        upper = differences.quantile(1 - alpha / 2)

        # Two-sided bootstrap sign-based p-value.
        p_positive = np.mean(differences <= 0)
        p_negative = np.mean(differences >= 0)

        p_value = min(
            1.0,
            2 * min(p_positive, p_negative),
        )

        records.append({
            "Feature": feature,
            "Q90_minus_Q10": (
                group[0.90] - group[0.10]
            ).mean(),
            "Bootstrap_Lower_95": lower,
            "Bootstrap_Upper_95": upper,
            "Bootstrap_P_approx": p_value,
            "CI_Excludes_Zero": lower > 0 or upper < 0,
            "Successful_Bootstraps": len(differences),
        })

    return pd.DataFrame(records)


def model_fit_summary(original):
    records = []

    for q, result in original.items():
        records.append({
            "Quantile": q,
            "Pseudo_R2": result.prsquared,
            "Objective": result.q,
            "N": int(result.nobs),
            "Iterations": result.iterations,
        })

    return pd.DataFrame(records)


def main():
    print("Loading macro-enhanced modeling panel...")
    df = load_data()

    print(f"Rows: {len(df)}")
    print(f"Years: {df['TID'].nunique()}")
    print(f"Categories: {df['ANVEND'].nunique()}")

    X, y, means, stds = build_design_matrix(df)

    print("\nFitting original quantile regressions...")
    original = fit_quantile_models(X, y)

    feature_names = X.columns.tolist()

    print("\nStarting year-cluster bootstrap...")
    bootstrap = bootstrap_coefficients(
        df,
        means,
        stds,
        feature_names,
    )

    if bootstrap.empty:
        raise RuntimeError(
            "No successful bootstrap regressions."
        )

    summary = coefficient_summary(
        original,
        bootstrap,
    )

    differences = quantile_difference_analysis(
        bootstrap
    )

    fit = model_fit_summary(original)

    summary.to_csv(
        OUT_DIR / "quantile_coefficients.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUT_DIR / "quantile_bootstrap_coefficients.csv",
        index=False,
    )

    differences.to_csv(
        OUT_DIR / "quantile_coefficient_differences.csv",
        index=False,
    )

    fit.to_csv(
        OUT_DIR / "quantile_model_fit.csv",
        index=False,
    )

    print("\n===== QUANTILE MODEL FIT =====")
    print(fit.round(4).to_string(index=False))

    print("\n===== QUANTILE COEFFICIENTS =====")
    print(
        summary.round(4).to_string(index=False)
    )

    print("\n===== Q90 - Q10 COEFFICIENT DIFFERENCES =====")
    print(
        differences.round(4).to_string(index=False)
    )

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
