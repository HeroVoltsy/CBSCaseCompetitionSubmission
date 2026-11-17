
#!/usr/bin/env python3

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from scipy.stats import chi2, t

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data/processed/causal_ml/construction_panel_macro_modeling.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data/processed/causal_ml/dml/heterogeneous_effects"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TREATMENT = "Permits_lag1"
OUTCOME = "Started"

NUMERIC_FEATURES = [
    "Started_lag2",
]

CATEGORY_FEATURE = "ANVEND"

N_SPLITS = 5
RANDOM_STATE = 42
N_BOOTSTRAP = 10000
BLOCK_LENGTHS = [2, 3]

GROUP_ORDER = [
    "Commercial_Industrial",
    "Public_Institutional",
    "Residential",
    "Other",
]

MODEL_NAMES = ["Ridge", "ExtraTrees"]


# ============================================================
# EXPLICIT CATEGORY TAXONOMY
# ============================================================

CATEGORY_GROUPS = {
    # Commercial and industrial
    "Factories, workshops, etc.": "Commercial_Industrial",
    "Other building used for production": "Commercial_Industrial",
    "Unspecified transport and trade": "Commercial_Industrial",
    "Office, trade, inventory, incl. public administration":
        "Commercial_Industrial",
    "Hotel, restaurant, hair dresser and other services":
        "Commercial_Industrial",

    # Public and institutional
    "Building for education and research (schools, laboratory etc.)":
        "Public_Institutional",
    "Building for hospital, home, maternity home etc.":
        "Public_Institutional",
    "Day-care institution": "Public_Institutional",
    "Non-specified welfare institutions": "Public_Institutional",
    "Student hostels": "Public_Institutional",
    "Library, church, museum etc.": "Public_Institutional",
    "Sports centres, club houses": "Public_Institutional",

    # Residential
    "Detached houses": "Residential",
    "Multi-dwelling houses": "Residential",
    "Farm houses": "Residential",
    "Other residential building": "Residential",
    "Residential buildings for communities": "Residential",
    "Terraced, linked or semi-detached houses": "Residential",
    "Weekend cottages": "Residential",
}


# ============================================================
# DATA PREPARATION
# ============================================================

def extract_year(series):
    """Extract a four-digit year from the source time column."""
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        return numeric.astype(int)

    extracted = series.astype(str).str.extract(r"((?:19|20)\d{2})")[0]
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def prepare_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "TID",
        CATEGORY_FEATURE,
        OUTCOME,
        TREATMENT,
        *NUMERIC_FEATURES,
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["YEAR"] = extract_year(df["TID"])

    if df["YEAR"].isna().any():
        bad_values = df.loc[df["YEAR"].isna(), "TID"].unique()
        raise ValueError(f"Could not extract years from TID: {bad_values}")

    df["YEAR"] = df["YEAR"].astype(int)

    numeric_columns = [
        OUTCOME,
        TREATMENT,
        *NUMERIC_FEATURES,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[CATEGORY_FEATURE] = df[CATEGORY_FEATURE].astype(str).str.strip()

    df = df.dropna(
        subset=[
            "YEAR",
            CATEGORY_FEATURE,
            OUTCOME,
            TREATMENT,
            *NUMERIC_FEATURES,
        ]
    ).copy()

    df = df.sort_values(
        ["YEAR", CATEGORY_FEATURE]
    ).reset_index(drop=True)

    df["Effect_Group"] = (
        df[CATEGORY_FEATURE]
        .map(CATEGORY_GROUPS)
        .fillna("Other")
    )

    return df


# ============================================================
# MAPPING AUDIT
# ============================================================

def audit_mapping(df):
    category_mapping = (
        df[[CATEGORY_FEATURE, "Effect_Group"]]
        .drop_duplicates()
        .sort_values(["Effect_Group", CATEGORY_FEATURE])
        .reset_index(drop=True)
    )

    category_mapping.to_csv(
        OUTPUT_DIR / "category_group_mapping.csv",
        index=False,
    )

    unmapped = sorted(
        set(df[CATEGORY_FEATURE].unique())
        - set(CATEGORY_GROUPS)
    )

    print("\n===== DATA AUDIT =====")
    print("Shape:", df.shape)
    print("Years:", df["YEAR"].nunique())
    print("Categories:", df[CATEGORY_FEATURE].nunique())

    print("\n===== CATEGORY GROUP MAPPING =====")
    print(category_mapping.to_string(index=False))

    counts = (
        df.groupby("Effect_Group")
        .agg(
            Categories=(CATEGORY_FEATURE, "nunique"),
            Observations=(OUTCOME, "size"),
            Years=("YEAR", "nunique"),
        )
        .reindex(GROUP_ORDER)
        .fillna(0)
    )

    counts[["Categories", "Observations", "Years"]] = (
        counts[["Categories", "Observations", "Years"]]
        .astype(int)
    )

    print("\n===== GROUP COUNTS =====")
    print(counts.to_string())

    print("\n===== MAPPING VALIDATION =====")
    print("Explicitly classified:", len(CATEGORY_GROUPS))
    print("Unmapped categories:", len(unmapped))
    print(unmapped)

    if df[CATEGORY_FEATURE].nunique() != 29:
        warnings.warn(
            "Expected 29 categories; observed "
            f"{df[CATEGORY_FEATURE].nunique()}."
        )

    if len(CATEGORY_GROUPS) != 19:
        raise AssertionError(
            "The explicit taxonomy should contain 19 categories."
        )

    if len(unmapped) != 10:
        warnings.warn(
            f"Expected 10 Other categories, but found {len(unmapped)}. "
            "Inspect category_group_mapping.csv."
        )

    expected_counts = {
        "Commercial_Industrial": 5,
        "Public_Institutional": 7,
        "Residential": 7,
        "Other": 10,
    }

    observed_counts = (
        df.groupby("Effect_Group")[CATEGORY_FEATURE]
        .nunique()
        .to_dict()
    )

    for group, expected in expected_counts.items():
        observed = observed_counts.get(group, 0)

        if observed != expected:
            warnings.warn(
                f"{group}: expected {expected} categories, "
                f"observed {observed}."
            )

    if (counts["Years"] == 0).any():
        raise ValueError(
            "At least one group has no observations. "
            "Cannot estimate all group-specific effects."
        )

    return category_mapping, counts


# ============================================================
# MODEL PIPELINES
# ============================================================

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES + ["YEAR_TREND", "YEAR_TREND_SQ"],
            ),
            (
                "category",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                [CATEGORY_FEATURE],
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def make_model(model_name):
    if model_name == "Ridge":
        estimator = Ridge(alpha=10.0)

    elif model_name == "ExtraTrees":
        estimator = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("estimator", estimator),
        ]
    )


# ============================================================
# YEAR-GROUPED CROSS-FITTING
# ============================================================

def cross_fit_residuals(df, model_name, targets):
    """
    Cross-fit nuisance predictions using GroupKFold by YEAR.

    All observations from a given year remain in the same fold.
    """

    X = df[
        NUMERIC_FEATURES
        + ["YEAR_TREND", "YEAR_TREND_SQ", CATEGORY_FEATURE]
    ].copy()

    years = df["YEAR"].to_numpy()

    splitter = GroupKFold(n_splits=N_SPLITS)

    residuals = {
        name: np.full(len(df), np.nan, dtype=float)
        for name in targets
    }

    base_model = make_model(model_name)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, groups=years),
        start=1,
    ):
        print(
            f"{model_name} | Fold {fold}/{N_SPLITS} | "
            f"Train={len(train_idx)} | Test={len(test_idx)}"
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        for target_name, target_values in targets.items():
            y_train = np.asarray(target_values)[train_idx]

            fitted_model = clone(base_model)
            fitted_model.fit(X_train, y_train)

            predictions = fitted_model.predict(X_test)

            residuals[target_name][test_idx] = (
                np.asarray(target_values)[test_idx] - predictions
            )

    for target_name, values in residuals.items():
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"Missing cross-fitted residuals for {target_name}."
            )

    return residuals


# ============================================================
# CLUSTER-ROBUST COVARIANCE
# ============================================================

def cluster_covariance(X, residual, clusters):
    """
    Cluster-robust sandwich covariance with a CR1 finite-sample
    correction and t reference distribution using G - 1 degrees
    of freedom.
    """

    X = np.asarray(X, dtype=float)
    residual = np.asarray(residual, dtype=float)
    clusters = np.asarray(clusters)

    n, k = X.shape
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    if n_clusters <= 1:
        raise ValueError("At least two clusters are required.")

    bread = np.linalg.pinv(X.T @ X)

    scores = X * residual[:, None]

    meat = np.zeros((k, k), dtype=float)

    for cluster in unique_clusters:
        cluster_score = scores[clusters == cluster].sum(axis=0)
        meat += np.outer(cluster_score, cluster_score)

    correction = (
        (n_clusters / (n_clusters - 1))
        * ((n - 1) / max(n - k, 1))
    )

    covariance = correction * bread @ meat @ bread
    covariance = (covariance + covariance.T) / 2

    return covariance, n_clusters


# ============================================================
# POOLED GROUP-SPECIFIC DML
# ============================================================

def estimate_group_effects(df, model_name):
    groups = [
        group
        for group in GROUP_ORDER
        if (df["Effect_Group"] == group).any()
    ]

    group_indicators = {
        group: (df["Effect_Group"] == group).astype(float).to_numpy()
        for group in groups
    }

    treatment = df[TREATMENT].to_numpy(dtype=float)
    outcome = df[OUTCOME].to_numpy(dtype=float)

    treatment_interactions = {
        f"D_x_{group}": treatment * indicator
        for group, indicator in group_indicators.items()
    }

    targets = {
        "outcome": outcome,
        **treatment_interactions,
    }

    residuals = cross_fit_residuals(
        df=df,
        model_name=model_name,
        targets=targets,
    )

    y_residual = residuals["outcome"]

    V = np.column_stack(
        [
            residuals[f"D_x_{group}"]
            for group in groups
        ]
    )

    residual_sd = V.std(axis=0, ddof=1)

    if np.any(residual_sd < 1e-8):
        bad_groups = [
            groups[i]
            for i, value in enumerate(residual_sd)
            if value < 1e-8
        ]

        raise ValueError(
            "Insufficient residualized treatment variation for: "
            + ", ".join(bad_groups)
        )

    rank = np.linalg.matrix_rank(V)

    if rank < V.shape[1]:
        raise ValueError(
            f"Residualized treatment matrix is rank deficient: "
            f"rank={rank}, columns={V.shape[1]}. "
            "Consider reducing the number of groups."
        )

    # Orthogonal-score regression, without an intercept.
    bread = np.linalg.pinv(V.T @ V)
    theta = bread @ V.T @ y_residual

    score_residual = y_residual - V @ theta

    covariance, n_clusters = cluster_covariance(
        X=V,
        residual=score_residual,
        clusters=df["YEAR"].to_numpy(),
    )

    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance), 0)
    )

    degrees_freedom = n_clusters - 1
    critical_value = t.ppf(0.975, degrees_freedom)

    p_values = 2 * t.sf(
        np.abs(theta / standard_errors),
        degrees_freedom,
    )

    group_rows = []

    for i, group in enumerate(groups):
        subset = df["Effect_Group"].to_numpy() == group

        group_rows.append(
            {
                "Model": model_name,
                "Group": group,
                "Categories": df.loc[
                    subset, CATEGORY_FEATURE
                ].nunique(),
                "Observations": int(subset.sum()),
                "Years": df.loc[subset, "YEAR"].nunique(),
                "Treatment_Mean": treatment[subset].mean(),
                "Treatment_SD": treatment[subset].std(ddof=1),
                "Residual_Treatment_SD": residual_sd[i],
                "Estimate": theta[i],
                "Cluster_SE": standard_errors[i],
                "CI_Lower_95": (
                    theta[i] - critical_value * standard_errors[i]
                ),
                "CI_Upper_95": (
                    theta[i] + critical_value * standard_errors[i]
                ),
                "P_Value": p_values[i],
            }
        )

    group_results = pd.DataFrame(group_rows)

    pairwise_results = calculate_pairwise_comparisons(
        theta=theta,
        covariance=covariance,
        groups=groups,
        model_name=model_name,
        degrees_freedom=degrees_freedom,
    )

    omnibus_result = calculate_omnibus_test(
        theta=theta,
        covariance=covariance,
        model_name=model_name,
    )

    bootstrap_results = circular_block_bootstrap(
        V=V,
        y_residual=y_residual,
        years=df["YEAR"].to_numpy(),
        groups=groups,
        model_name=model_name,
    )

    oof = pd.DataFrame(
        {
            "YEAR": df["YEAR"].to_numpy(),
            CATEGORY_FEATURE: df[CATEGORY_FEATURE].to_numpy(),
            "Effect_Group": df["Effect_Group"].to_numpy(),
            "Outcome": outcome,
            "Treatment": treatment,
            "Outcome_Residual": y_residual,
        }
    )

    for i, group in enumerate(groups):
        oof[f"Treatment_Residual_{group}"] = V[:, i]

    return (
        group_results,
        pairwise_results,
        omnibus_result,
        bootstrap_results,
        oof,
    )


# ============================================================
# PAIRWISE GROUP COMPARISONS
# ============================================================

def holm_adjust(p_values):
    """Holm step-down family-wise error correction."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)

    running_max = 0.0

    for rank, index in enumerate(order):
        value = (n - rank) * p_values[index]
        running_max = max(running_max, value)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def calculate_pairwise_comparisons(
    theta,
    covariance,
    groups,
    model_name,
    degrees_freedom,
):
    rows = []
    contrasts = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            contrast = np.zeros(len(groups))
            contrast[i] = 1.0
            contrast[j] = -1.0

            difference = float(contrast @ theta)
            variance = float(contrast @ covariance @ contrast)
            se = np.sqrt(max(variance, 0.0))

            if se <= 1e-12:
                p_value = np.nan
                ci_lower = np.nan
                ci_upper = np.nan
            else:
                statistic = difference / se
                p_value = 2 * t.sf(
                    abs(statistic),
                    degrees_freedom,
                )

                critical = t.ppf(
                    0.975,
                    degrees_freedom,
                )

                ci_lower = difference - critical * se
                ci_upper = difference + critical * se

            rows.append(
                {
                    "Group_1": groups[i],
                    "Group_2": groups[j],
                    "Difference": difference,
                    "Cluster_SE": se,
                    "CI_Lower_95": ci_lower,
                    "CI_Upper_95": ci_upper,
                    "P_Value": p_value,
                    "Model": model_name,
                }
            )

    result = pd.DataFrame(rows)

    valid = result["P_Value"].notna()

    result["Holm_P_Value"] = np.nan

    if valid.any():
        result.loc[valid, "Holm_P_Value"] = holm_adjust(
            result.loc[valid, "P_Value"].to_numpy()
        )

    return result


# ============================================================
# OMNIBUS HETEROGENEITY TEST
# ============================================================


def calculate_omnibus_test(theta, covariance, model_name):
    """
    Test equality of all group-specific coefficients.

    H0: theta_1 = theta_2 = ... = theta_k

    This is a heterogeneity test, unlike a test of whether
    all group coefficients jointly equal zero.
    """

    theta = np.asarray(theta, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    k = len(theta)

    if k < 2:
        raise ValueError("At least two groups are required.")

    # Compare every group against the first group.
    R = np.zeros((k - 1, k))

    for i in range(1, k):
        R[i - 1, 0] = -1.0
        R[i - 1, i] = 1.0

    differences = R @ theta
    contrast_covariance = R @ covariance @ R.T

    statistic = float(
        differences.T
        @ np.linalg.pinv(contrast_covariance)
        @ differences
    )

    degrees_freedom = np.linalg.matrix_rank(
        contrast_covariance
    )

    p_value = chi2.sf(statistic, degrees_freedom)

    return {
        "Wald_Statistic": statistic,
        "DF": degrees_freedom,
        "P_Value": p_value,
        "Null_Hypothesis": "All group coefficients are equal",
        "Model": model_name,
    }



# ============================================================
# CIRCULAR MOVING-BLOCK BOOTSTRAP
# ============================================================

def circular_block_bootstrap(
    V,
    y_residual,
    years,
    groups,
    model_name,
):
    """
    Resample contiguous annual blocks using annual score
    contributions. Nuisance models are not refitted in bootstrap
    replications; intervals are conditional on cross-fitted
    nuisance predictions.
    """

    unique_years = np.sort(np.unique(years))
    n_years = len(unique_years)
    n_groups = V.shape[1]

    year_positions = {
        year: np.where(years == year)[0]
        for year in unique_years
    }

    annual_A = np.zeros(
        (n_years, n_groups, n_groups),
        dtype=float,
    )

    annual_b = np.zeros(
        (n_years, n_groups),
        dtype=float,
    )

    for i, year in enumerate(unique_years):
        indices = year_positions[year]

        V_year = V[indices]
        y_year = y_residual[indices]

        annual_A[i] = V_year.T @ V_year
        annual_b[i] = V_year.T @ y_year

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for block_length in BLOCK_LENGTHS:
        estimates = []

        for _ in range(N_BOOTSTRAP):
            sampled_indices = []

            while len(sampled_indices) < n_years:
                start = rng.integers(0, n_years)

                block = [
                    (start + offset) % n_years
                    for offset in range(block_length)
                ]

                sampled_indices.extend(block)

            sampled_indices = sampled_indices[:n_years]

            A_boot = annual_A[sampled_indices].sum(axis=0)
            b_boot = annual_b[sampled_indices].sum(axis=0)

            theta_boot = np.linalg.pinv(A_boot) @ b_boot

            if np.isfinite(theta_boot).all():
                estimates.append(theta_boot)

        estimates = np.asarray(estimates, dtype=float)

        if len(estimates) == 0:
            warnings.warn(
                f"No valid bootstrap estimates for block "
                f"length {block_length}."
            )
            continue

        print(
            f"Block {block_length}: "
            f"{len(estimates)} valid replications"
        )

        for i, group in enumerate(groups):
            lower, upper = np.quantile(
                estimates[:, i],
                [0.025, 0.975],
            )

            rows.append(
                {
                    "Model": model_name,
                    "Group": group,
                    "Block_Length": block_length,
                    "Valid_Replications": len(estimates),
                    "Bootstrap_Mean": estimates[:, i].mean(),
                    "Bootstrap_SD": estimates[:, i].std(ddof=1),
                    "Bootstrap_CI_Lower_95": lower,
                    "Bootstrap_CI_Upper_95": upper,
                    "Positive_Estimate_Share": (
                        estimates[:, i] > 0
                    ).mean(),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    df = prepare_data()

    # Deterministic linear time controls.
    min_year = df["YEAR"].min()

    df["YEAR_TREND"] = df["YEAR"] - min_year
    df["YEAR_TREND_SQ"] = df["YEAR_TREND"] ** 2

    category_mapping, group_counts = audit_mapping(df)

    if df["YEAR"].nunique() < N_SPLITS:
        raise ValueError(
            f"Need at least {N_SPLITS} distinct years for GroupKFold."
        )

    all_group_results = []
    all_pairwise_results = []
    all_omnibus_results = []
    all_bootstrap_results = []
    all_oof_results = []

    for model_name in MODEL_NAMES:
        print(f"\n===== HETEROGENEITY: {model_name} =====")

        (
            group_results,
            pairwise_results,
            omnibus_result,
            bootstrap_results,
            oof_results,
        ) = estimate_group_effects(
            df=df,
            model_name=model_name,
        )

        print("\n===== GROUP EFFECTS =====")
        print(group_results.to_string(index=False))

        print("\n===== PAIRWISE COMPARISONS =====")
        print(pairwise_results.to_string(index=False))

        print("\n===== OMNIBUS HETEROGENEITY =====")
        print(omnibus_result)

        all_group_results.append(group_results)
        all_pairwise_results.append(pairwise_results)
        all_omnibus_results.append(omnibus_result)
        all_bootstrap_results.append(bootstrap_results)
        all_oof_results.append(oof_results)

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    final_groups = pd.concat(
        all_group_results,
        ignore_index=True,
    )

    final_pairwise = pd.concat(
        all_pairwise_results,
        ignore_index=True,
    )

    final_omnibus = pd.DataFrame(all_omnibus_results)

    final_bootstrap = pd.concat(
        all_bootstrap_results,
        ignore_index=True,
    )

    final_oof = pd.concat(
        all_oof_results,
        ignore_index=True,
    )

    final_groups.to_csv(
        OUTPUT_DIR / "group_effect_estimates.csv",
        index=False,
    )

    final_pairwise.to_csv(
        OUTPUT_DIR / "pairwise_group_comparisons.csv",
        index=False,
    )

    final_omnibus.to_csv(
        OUTPUT_DIR / "omnibus_heterogeneity_tests.csv",
        index=False,
    )

    final_bootstrap.to_csv(
        OUTPUT_DIR / "group_block_bootstrap.csv",
        index=False,
    )

    final_oof.to_csv(
        OUTPUT_DIR / "heterogeneity_oof_residuals.csv",
        index=False,
    )

    print("\n===== FINAL GROUP EFFECTS =====")
    print(final_groups.to_string(index=False))

    print("\n===== FINAL OMNIBUS TESTS =====")
    print(final_omnibus.to_string(index=False))

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
