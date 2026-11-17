
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "causal_ml"
OUT_DIR = DATA_DIR / "cpcv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_FILE = DATA_DIR / "construction_panel_modeling.csv"
MACRO_FILE = DATA_DIR / "construction_panel_macro_modeling.csv"

FIRST_YEAR = 2003
LAST_YEAR = 2025

N_GROUPS = 6
N_TEST_GROUPS = 2
EMBARGO_YEARS = 1

RANDOM_STATE = 42
N_JOBS = -1

TARGET = "Started"

NUMERIC_FEATURES = [
    "Started_lag1",
    "Started_lag2",
    "Permits_lag1",
    "Completed_lag1",
    "Under construction_lag1",
]

MACRO_FEATURES = [
    "gdp_growth_pct_lag1",
    "corporate_lending_rate_pct_lag1",
    "household_lending_rate_pct_lag1",
    "inflation_pct_lag1",
]

CATEGORICAL_FEATURES = ["ANVEND"]

MODEL_NAMES = [
    "Persistence",
    "Ridge",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def load_panel(path):
    df = pd.read_csv(path)

    required = [
        "TID",
        "ANVEND",
        TARGET,
        *NUMERIC_FEATURES,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {missing}"
        )

    df["TID"] = pd.to_numeric(df["TID"], errors="coerce")
    df = df.dropna(subset=["TID", TARGET, "ANVEND"])

    df["TID"] = df["TID"].astype(int)
    df["ANVEND"] = df["ANVEND"].astype(str)

    df = df[
        df["TID"].between(FIRST_YEAR, LAST_YEAR)
    ].copy()

    df = df.sort_values(["TID", "ANVEND"]).reset_index(drop=True)

    return df


def make_temporal_groups(years, n_groups):
    years = np.array(sorted(years))

    if len(years) < n_groups:
        raise ValueError("Not enough years for requested CPCV groups.")

    groups = np.array_split(years, n_groups)

    return {
        i: set(group.tolist())
        for i, group in enumerate(groups)
    }


def make_cpcv_splits(df):
    years = sorted(df["TID"].unique())

    groups = make_temporal_groups(years, N_GROUPS)

    splits = []

    for split_id, test_group_ids in enumerate(
        combinations(groups.keys(), N_TEST_GROUPS),
        start=1,
    ):
        test_years = set()

        for group_id in test_group_ids:
            test_years.update(groups[group_id])

        embargo_years = set()

        for test_year in test_years:
            for offset in range(-EMBARGO_YEARS, EMBARGO_YEARS + 1):
                embargo_years.add(test_year + offset)

        train_mask = ~df["TID"].isin(embargo_years)
        test_mask = df["TID"].isin(test_years)

        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        splits.append({
            "split_id": split_id,
            "test_groups": ",".join(map(str, test_group_ids)),
            "test_years": sorted(test_years),
            "train_idx": train_idx,
            "test_idx": test_idx,
        })

    return groups, splits


def make_preprocessor(numeric_features, scale_numeric=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_models(numeric_features):
    models = {}

    models["Ridge"] = Pipeline([
        (
            "preprocessor",
            make_preprocessor(numeric_features, scale_numeric=True),
        ),
        ("model", Ridge(alpha=10.0)),
    ])

    models["RandomForest"] = Pipeline([
        (
            "preprocessor",
            make_preprocessor(numeric_features),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
        ),
    ])

    models["ExtraTrees"] = Pipeline([
        (
            "preprocessor",
            make_preprocessor(numeric_features),
        ),
        (
            "model",
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
        ),
    ])

    models["XGBoost"] = Pipeline([
        (
            "preprocessor",
            make_preprocessor(numeric_features),
        ),
        (
            "model",
            XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbosity=0,
            ),
        ),
    ])

    models["LightGBM"] = Pipeline([
        (
            "preprocessor",
            make_preprocessor(numeric_features),
        ),
        (
            "model",
            LGBMRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=10,
                reg_lambda=5.0,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                verbosity=-1,
            ),
        ),
    ])

    return models


def build_catboost():
    return CatBoostRegressor(
        iterations=300,
        depth=5,
        learning_rate=0.03,
        loss_function="RMSE",
        l2_leaf_reg=5.0,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=N_JOBS,
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denominator = np.sum(np.abs(y_true))
    wape = (
        np.sum(np.abs(y_true - y_pred)) / denominator * 100
        if denominator > 0
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_pct": wape,
        "R2": r2_score(y_true, y_pred),
        "Bias": np.mean(y_pred - y_true),
    }


def run_benchmark(df, dataset_name, numeric_features, splits):
    prediction_records = []
    metric_records = []

    model_pipelines = build_models(numeric_features)

    for split in splits:
        split_id = split["split_id"]

        train = df.iloc[split["train_idx"]].copy()
        test = df.iloc[split["test_idx"]].copy()

        X_train = train[numeric_features + CATEGORICAL_FEATURES]
        X_test = test[numeric_features + CATEGORICAL_FEATURES]

        y_train = train[TARGET].astype(float)
        y_test = test[TARGET].astype(float)

        predictions = {}

        predictions["Persistence"] = test["Started_lag1"].to_numpy()

        for model_name, model in model_pipelines.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            predictions[model_name] = fitted.predict(X_test)

        cat_train = X_train.copy()
        cat_test = X_test.copy()

        for col in CATEGORICAL_FEATURES:
            cat_train[col] = cat_train[col].astype(str)
            cat_test[col] = cat_test[col].astype(str)

        cat_model = build_catboost()

        cat_model.fit(
            cat_train,
            y_train,
            cat_features=CATEGORICAL_FEATURES,
        )

        predictions["CatBoost"] = cat_model.predict(cat_test)

        for model_name, y_pred in predictions.items():
            y_pred = np.asarray(y_pred, dtype=float)

            metrics = calculate_metrics(y_test, y_pred)

            metric_records.append({
                "Dataset": dataset_name,
                "Split": split_id,
                "Test_Groups": split["test_groups"],
                "Test_Years": ",".join(map(str, split["test_years"])),
                "Model": model_name,
                "Train_N": len(train),
                "Test_N": len(test),
                **metrics,
            })

            output = test[
                ["TID", "ANVEND", TARGET]
            ].copy()

            output["Dataset"] = dataset_name
            output["Split"] = split_id
            output["Test_Groups"] = split["test_groups"]
            output["Model"] = model_name
            output["Prediction"] = y_pred
            output["Residual"] = output[TARGET] - output["Prediction"]
            output["Absolute_Error"] = np.abs(output["Residual"])

            prediction_records.append(output)

        print(
            f"{dataset_name} | Split {split_id}/{len(splits)} "
            f"| train={len(train)} test={len(test)}"
        )

    return (
        pd.concat(prediction_records, ignore_index=True),
        pd.DataFrame(metric_records),
    )


def summarize_metrics(metrics):
    summary = (
        metrics.groupby(["Dataset", "Model"])
        .agg(
            Splits=("Split", "nunique"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            WAPE_mean=("WAPE_pct", "mean"),
            WAPE_std=("WAPE_pct", "std"),
            R2_mean=("R2", "mean"),
            Bias_mean=("Bias", "mean"),
        )
        .reset_index()
    )

    return summary.sort_values(
        ["Dataset", "RMSE_mean"]
    )


def category_metrics(predictions):
    records = []

    for keys, group in predictions.groupby(
        ["Dataset", "Model", "ANVEND"]
    ):
        dataset, model, category = keys

        metric = calculate_metrics(
            group[TARGET],
            group["Prediction"],
        )

        records.append({
            "Dataset": dataset,
            "Model": model,
            "ANVEND": category,
            "Observations": len(group),
            **metric,
        })

    return pd.DataFrame(records)


def paired_split_comparison(metrics):
    records = []

    for dataset, data in metrics.groupby("Dataset"):
        pivot = data.pivot(
            index="Split",
            columns="Model",
            values="RMSE",
        )

        if "Persistence" not in pivot.columns:
            continue

        for model in pivot.columns:
            if model == "Persistence":
                continue

            valid = pivot[["Persistence", model]].dropna()

            differences = (
                valid["Persistence"] - valid[model]
            )

            records.append({
                "Dataset": dataset,
                "Model": model,
                "Paired_Splits": len(differences),
                "Mean_RMSE_Improvement_vs_Persistence": differences.mean(),
                "Median_RMSE_Improvement_vs_Persistence": differences.median(),
                "Improvement_Share": (differences > 0).mean(),
                "Mean_Relative_Improvement_pct": (
                    differences / valid["Persistence"] * 100
                ).mean(),
            })

    return pd.DataFrame(records)


def macro_incremental_value(metrics):
    pivot = metrics.pivot_table(
        index=["Split", "Model"],
        columns="Dataset",
        values=["RMSE", "WAPE_pct", "MAE"],
    )

    records = []

    for model in MODEL_NAMES:
        try:
            full_rmse = pivot.loc[(slice(None), model), ("RMSE", "Full_History")]
            macro_rmse = pivot.loc[(slice(None), model), ("RMSE", "Macro_Enhanced")]

            full_rmse.index = full_rmse.index.get_level_values("Split")
            macro_rmse.index = macro_rmse.index.get_level_values("Split")

            paired = pd.concat(
                [
                    full_rmse.rename("Full_RMSE"),
                    macro_rmse.rename("Macro_RMSE"),
                ],
                axis=1,
            ).dropna()

            difference = paired["Full_RMSE"] - paired["Macro_RMSE"]

            records.append({
                "Model": model,
                "Paired_Splits": len(paired),
                "Mean_RMSE_Improvement_from_Macro": difference.mean(),
                "Median_RMSE_Improvement_from_Macro": difference.median(),
                "Improvement_Share": (difference > 0).mean(),
                "Mean_Relative_RMSE_Improvement_pct": (
                    difference / paired["Full_RMSE"] * 100
                ).mean(),
            })

        except KeyError:
            continue

    return pd.DataFrame(records)


def main():
    print("Loading datasets...")

    full = load_panel(FULL_FILE)
    macro = load_panel(MACRO_FILE)

    common_years = sorted(
        set(full["TID"].unique())
        & set(macro["TID"].unique())
    )

    common_categories = sorted(
        set(full["ANVEND"].unique())
        & set(macro["ANVEND"].unique())
    )

    full = full[
        full["TID"].isin(common_years)
        & full["ANVEND"].isin(common_categories)
    ].copy()

    macro = macro[
        macro["TID"].isin(common_years)
        & macro["ANVEND"].isin(common_categories)
    ].copy()

    macro_missing = [
        c for c in MACRO_FEATURES
        if c not in macro.columns
    ]

    if macro_missing:
        raise ValueError(
            f"Missing macro features: {macro_missing}"
        )

    full = full.dropna(
        subset=NUMERIC_FEATURES + [TARGET]
    ).reset_index(drop=True)

    macro = macro.dropna(
        subset=NUMERIC_FEATURES + MACRO_FEATURES + [TARGET]
    ).reset_index(drop=True)

    full = full[
        full["TID"].isin(common_years)
    ].copy()

    macro = macro[
        macro["TID"].isin(common_years)
    ].copy()

    if not full.groupby("TID")["ANVEND"].nunique().eq(
        len(common_categories)
    ).all():
        raise ValueError("Full-history panel is not balanced by year/category.")

    if not macro.groupby("TID")["ANVEND"].nunique().eq(
        len(common_categories)
    ).all():
        raise ValueError("Macro panel is not balanced by year/category.")

    if not full[["TID", "ANVEND"]].equals(
        macro[["TID", "ANVEND"]]
    ):
        raise ValueError(
            "Datasets do not have identical year/category observations."
        )

    print(f"Common years: {min(common_years)}–{max(common_years)}")
    print(f"Years: {len(common_years)}")
    print(f"Categories: {len(common_categories)}")
    print(f"Observations per dataset: {len(full)}")

    groups, splits = make_cpcv_splits(full)

    print("\nCPCV CONFIGURATION")
    for group_id, years in groups.items():
        print(f"Group {group_id}: {min(years)}–{max(years)}")

    print(f"Combinatorial splits: {len(splits)}")
    print(f"Embargo: {EMBARGO_YEARS} year(s)")

    full_predictions, full_metrics = run_benchmark(
        full,
        "Full_History",
        NUMERIC_FEATURES,
        splits,
    )

    macro_predictions, macro_metrics = run_benchmark(
        macro,
        "Macro_Enhanced",
        NUMERIC_FEATURES + MACRO_FEATURES,
        splits,
    )

    predictions = pd.concat(
        [full_predictions, macro_predictions],
        ignore_index=True,
    )

    metrics = pd.concat(
        [full_metrics, macro_metrics],
        ignore_index=True,
    )

    summary = summarize_metrics(metrics)
    categories = category_metrics(predictions)
    paired = paired_split_comparison(metrics)
    macro_value = macro_incremental_value(metrics)

    predictions.to_csv(
        OUT_DIR / "cpcv_predictions.csv",
        index=False,
    )

    metrics.to_csv(
        OUT_DIR / "cpcv_split_metrics.csv",
        index=False,
    )

    summary.to_csv(
        OUT_DIR / "cpcv_model_comparison.csv",
        index=False,
    )

    categories.to_csv(
        OUT_DIR / "cpcv_category_metrics.csv",
        index=False,
    )

    paired.to_csv(
        OUT_DIR / "cpcv_paired_comparison.csv",
        index=False,
    )

    macro_value.to_csv(
        OUT_DIR / "cpcv_macro_incremental_value.csv",
        index=False,
    )

    print("\n===== CPCV MODEL COMPARISON =====")
    print(summary.round(4).to_string(index=False))

    print("\n===== PAIRED IMPROVEMENT VS PERSISTENCE =====")
    print(paired.round(4).to_string(index=False))

    print("\n===== INCREMENTAL VALUE OF MACRO FEATURES =====")
    print(macro_value.round(4).to_string(index=False))

    print("\nSaved outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
