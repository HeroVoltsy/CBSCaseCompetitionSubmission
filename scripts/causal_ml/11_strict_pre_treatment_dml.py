
from pathlib import Path

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
    ROOT / "data/processed/causal_ml/"
    / "construction_panel_macro_modeling.csv"
)

OUT = (
    ROOT / "data/processed/causal_ml/"
    / "dml/strict_pre_treatment"
)

OUT.mkdir(parents=True, exist_ok=True)

YEAR = "TID"
CATEGORY = "ANVEND"
TREATMENT = "Permits_lag1"
TARGET = "Started"

NUMERIC_CONTROLS = [
    "Started_lag2",
    "YEAR_TREND",
    "YEAR_TREND_SQ",
]

N_SPLITS = 5
RANDOM_STATE = 42


def make_pipeline(model_name):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    if model_name == "Ridge":
        numeric_pipeline.steps.append(
            ("scaler", StandardScaler())
        )

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_CONTROLS),
        ("category", categorical_pipeline, [CATEGORY]),
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
        ("preprocessing", preprocessing),
        ("estimator", estimator),
    ])


def cluster_inference(residual_y, residual_t, years):
    denominator = np.sum(residual_t ** 2)

    theta = np.sum(
        residual_t * residual_y
    ) / denominator

    scores = residual_t * (
        residual_y - theta * residual_t
    )

    score_df = pd.DataFrame({
        "year": years,
        "score": scores,
    })

    cluster_scores = (
        score_df.groupby("year")["score"]
        .sum()
        .to_numpy()
    )

    n_clusters = len(cluster_scores)

    variance = (
        n_clusters / (n_clusters - 1)
        * np.sum(cluster_scores ** 2)
        / denominator ** 2
    )

    se = np.sqrt(variance)
    df = n_clusters - 1

    critical = student_t.ppf(0.975, df=df)

    statistic = theta / se

    p_value = 2 * student_t.sf(
        abs(statistic),
        df=df,
    )

    return {
        "Estimate": theta,
        "Cluster_SE": se,
        "CI_Lower_95": theta - critical * se,
        "CI_Upper_95": theta + critical * se,
        "T_Statistic": statistic,
        "P_Value": p_value,
        "N_Clusters": n_clusters,
    }


def run_model(df, model_name):
    features = NUMERIC_CONTROLS + [CATEGORY]

    X = df[features].copy()

    y = df[TARGET].to_numpy(dtype=float)
    treatment = df[TREATMENT].to_numpy(dtype=float)
    years = df[YEAR].to_numpy()

    y_hat = np.full(len(df), np.nan)
    treatment_hat = np.full(len(df), np.nan)

    splitter = GroupKFold(n_splits=N_SPLITS)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=years),
        start=1,
    ):
        y_model = make_pipeline(model_name)
        t_model = make_pipeline(model_name)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_model.fit(X_train, y[train_idx])
        t_model.fit(X_train, treatment[train_idx])

        y_hat[test_idx] = y_model.predict(X_test)
        treatment_hat[test_idx] = (
            t_model.predict(X_test)
        )

        print(
            f"{model_name} | Fold {fold}/{N_SPLITS} | "
            f"Train={len(train_idx)} | "
            f"Test={len(test_idx)}"
        )

    if np.isnan(y_hat).any() or np.isnan(treatment_hat).any():
        raise RuntimeError("Incomplete cross-fitted predictions.")

    residual_y = y - y_hat
    residual_t = treatment - treatment_hat

    result = cluster_inference(
        residual_y,
        residual_t,
        years,
    )

    result.update({
        "Model": model_name,
        "N": len(df),
        "N_Years": df[YEAR].nunique(),
        "Treatment_SD": np.std(treatment, ddof=1),
        "Residual_Treatment_SD": np.std(
            residual_t,
            ddof=1,
        ),
        "Treatment_Nuisance_R2": (
            1 - np.var(residual_t, ddof=1)
            / np.var(treatment, ddof=1)
        ),
        "Outcome_Residual_SD": np.std(
            residual_y,
            ddof=1,
        ),
    })

    predictions = pd.DataFrame({
        "Year": years,
        "Category": df[CATEGORY].to_numpy(),
        "Observed_Started": y,
        "Observed_Permits_lag1": treatment,
        "Predicted_Started": y_hat,
        "Predicted_Permits": treatment_hat,
        "Residual_Started": residual_y,
        "Residual_Permits": residual_t,
        "Orthogonal_Score": (
            residual_t
            * (
                residual_y
                - result["Estimate"] * residual_t
            )
        ),
        "Model": model_name,
    })

    return result, predictions


def main():
    df = pd.read_csv(INPUT)

    required = [
        YEAR,
        CATEGORY,
        TARGET,
        TREATMENT,
        "Started_lag2",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()

    for column in [
        YEAR,
        TARGET,
        TREATMENT,
        "Started_lag2",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(subset=required).copy()

    df[YEAR] = df[YEAR].astype(int)
    df[CATEGORY] = df[CATEGORY].astype(str)

    # Center the year trend to improve numerical stability.
    df["YEAR_TREND"] = (
        df[YEAR] - df[YEAR].mean()
    )

    df["YEAR_TREND_SQ"] = (
        df["YEAR_TREND"] ** 2
    )

    df = df.reset_index(drop=True)

    print("===== STRICT PRE-TREATMENT AUDIT =====")
    print("Shape:", df.shape)
    print("Years:", df[YEAR].min(), "-", df[YEAR].max())
    print("Unique years:", df[YEAR].nunique())
    print("Categories:", df[CATEGORY].nunique())
    print("Controls:", NUMERIC_CONTROLS + [CATEGORY])

    results = []
    predictions = []

    for model_name in ["Ridge", "ExtraTrees"]:
        print(f"\n===== {model_name} =====")

        result, prediction = run_model(
            df,
            model_name,
        )

        results.append(result)
        predictions.append(prediction)

        print(pd.Series(result).to_string())

    result_df = pd.DataFrame(results)

    prediction_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    result_df.to_csv(
        OUT / "strict_dml_estimates.csv",
        index=False,
    )

    prediction_df.to_csv(
        OUT / "strict_dml_predictions.csv",
        index=False,
    )

    print("\n===== STRICT DML SUMMARY =====")
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
                "Residual_Treatment_SD",
                "N_Years",
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()
