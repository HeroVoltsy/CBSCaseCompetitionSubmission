
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data/processed/causal_ml/"
    / "construction_panel_macro_modeling.csv"
)

OUT = (
    ROOT
    / "data/processed/causal_ml/"
    / "dml"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGET = "Started"
TREATMENT = "Permits_lag1"
YEAR = "TID"
CATEGORY = "ANVEND"

CONTROLS = [
    "Started_lag1",
    "Started_lag2",
    "Completed_lag1",
    "Under construction_lag1",
    "gdp_growth_pct_lag1",
    "corporate_lending_rate_pct_lag1",
    "household_lending_rate_pct_lag1",
    "inflation_pct_lag1",
]

N_SPLITS = 5
RANDOM_STATE = 42


def make_pipeline(model_name, numeric_features):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    if model_name == "Ridge":
        numeric.steps.append(
            ("scaler", StandardScaler())
        )

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("category", categorical, [CATEGORY]),
    ])

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
        raise ValueError(model_name)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator),
    ])


def clustered_inference(
    residual_y,
    residual_t,
    years,
):
    denominator = np.sum(residual_t ** 2)

    theta = (
        np.sum(residual_t * residual_y)
        / denominator
    )

    scores = residual_t * (
        residual_y - theta * residual_t
    )

    score_frame = pd.DataFrame({
        "year": years,
        "score": scores,
    })

    cluster_scores = (
        score_frame
        .groupby("year")["score"]
        .sum()
        .to_numpy()
    )

    n_clusters = len(cluster_scores)

    if n_clusters < 2:
        raise ValueError("Insufficient year clusters.")

    variance = (
        n_clusters / (n_clusters - 1)
        * np.sum(cluster_scores ** 2)
        / denominator ** 2
    )

    standard_error = np.sqrt(variance)

    degrees_freedom = n_clusters - 1

    critical = student_t.ppf(
        0.975,
        df=degrees_freedom,
    )

    t_statistic = theta / standard_error

    p_value = 2 * student_t.sf(
        abs(t_statistic),
        df=degrees_freedom,
    )

    return {
        "Estimate": theta,
        "Cluster_SE": standard_error,
        "CI_Lower_95": theta - critical * standard_error,
        "CI_Upper_95": theta + critical * standard_error,
        "T_Statistic": t_statistic,
        "P_Value": p_value,
        "N_Clusters": n_clusters,
    }


def run_dml(df, model_name):
    numeric_features = CONTROLS + [YEAR]

    X = df[numeric_features + [CATEGORY]].copy()

    y = df[TARGET].to_numpy(dtype=float)
    treatment = df[TREATMENT].to_numpy(dtype=float)

    groups = df[YEAR].to_numpy()

    y_hat = np.full(len(df), np.nan)
    treatment_hat = np.full(len(df), np.nan)

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_model = make_pipeline(
            model_name,
            numeric_features,
        )

        treatment_model = make_pipeline(
            model_name,
            numeric_features,
        )

        y_model.fit(
            X_train,
            y[train_idx],
        )

        treatment_model.fit(
            X_train,
            treatment[train_idx],
        )

        y_hat[test_idx] = y_model.predict(X_test)

        treatment_hat[test_idx] = (
            treatment_model.predict(X_test)
        )

        print(
            f"{model_name} | Fold {fold}/{N_SPLITS} | "
            f"Train={len(train_idx)} | "
            f"Test={len(test_idx)} | "
            f"Test years={len(np.unique(groups[test_idx]))}"
        )

    if np.isnan(y_hat).any() or np.isnan(treatment_hat).any():
        raise RuntimeError("Missing out-of-fold predictions.")

    residual_y = y - y_hat
    residual_t = treatment - treatment_hat

    inference = clustered_inference(
        residual_y,
        residual_t,
        groups,
    )

    treatment_variance = np.var(treatment, ddof=1)
    residual_t_variance = np.var(residual_t, ddof=1)

    treatment_r2 = (
        1 - residual_t_variance / treatment_variance
        if treatment_variance > 0
        else np.nan
    )

    inference.update({
        "Model": model_name,
        "N": len(df),
        "N_Years": df[YEAR].nunique(),
        "Treatment_SD": np.std(treatment, ddof=1),
        "Residual_Treatment_SD": np.std(
            residual_t,
            ddof=1,
        ),
        "Treatment_Nuisance_R2": treatment_r2,
        "Outcome_Residual_SD": np.std(
            residual_y,
            ddof=1,
        ),
    })

    predictions = pd.DataFrame({
        "Year": groups,
        "Category": df[CATEGORY].to_numpy(),
        "Observed_Started": y,
        "Observed_Permits_lag1": treatment,
        "Predicted_Outcome": y_hat,
        "Predicted_Treatment": treatment_hat,
        "Residual_Outcome": residual_y,
        "Residual_Treatment": residual_t,
        "Orthogonal_Score": (
            residual_t
            * (residual_y - inference["Estimate"] * residual_t)
        ),
        "Model": model_name,
    })

    return inference, predictions


def main():
    df = pd.read_csv(INPUT)

    required = [
        YEAR,
        CATEGORY,
        TARGET,
        TREATMENT,
    ] + CONTROLS

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[required].copy()

    for column in [YEAR, TARGET, TREATMENT] + CONTROLS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    df[YEAR] = df[YEAR].astype(int)
    df[CATEGORY] = df[CATEGORY].astype(str)

    print("===== DATA AUDIT =====")
    print("Shape:", df.shape)
    print("Years:", df[YEAR].min(), "-", df[YEAR].max())
    print("Unique years:", df[YEAR].nunique())
    print("Categories:", df[CATEGORY].nunique())
    print("Treatment mean:", round(df[TREATMENT].mean(), 3))
    print("Treatment SD:", round(df[TREATMENT].std(), 3))
    print("Outcome mean:", round(df[TARGET].mean(), 3))

    if df[YEAR].nunique() < N_SPLITS:
        raise ValueError("Not enough years for grouped cross-fitting.")

    results = []
    prediction_frames = []

    for model_name in ["Ridge", "ExtraTrees"]:
        print(f"\n===== DML: {model_name} =====")

        result, predictions = run_dml(
            df,
            model_name,
        )

        results.append(result)
        prediction_frames.append(predictions)

        print(pd.Series(result).to_string())

    result_df = pd.DataFrame(results)

    prediction_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    result_df.to_csv(
        OUT / "dml_estimates.csv",
        index=False,
    )

    prediction_df.to_csv(
        OUT / "dml_oof_predictions.csv",
        index=False,
    )

    print("\n===== DML ESTIMATES =====")
    print(
        result_df[
            [
                "Model",
                "Estimate",
                "Cluster_SE",
                "CI_Lower_95",
                "CI_Upper_95",
                "P_Value",
                "Treatment_Nuisance_R2",
                "N_Years",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )
    main()
