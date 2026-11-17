
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

BASE = Path("data/processed/causal_ml")
OUTPUT = BASE
OUTPUT.mkdir(parents=True, exist_ok=True)

FIRST_TRAIN_YEAR = 2003
FIRST_TEST_YEAR = 2013
LAST_YEAR = 2025

DATASETS = {
    "Full_History_Common": BASE / "construction_panel_modeling.csv",
    "Macro_Enhanced_Common": BASE / "construction_panel_macro_modeling.csv"
}


def build_models():
    return {
        "Ridge": Ridge(alpha=10.0),

        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    }


def build_preprocessor(categorical, numeric, scale_numeric=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer([
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        (
            "numeric",
            Pipeline(numeric_steps),
            numeric
        )
    ])


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "N": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "WAPE_pct": (
            np.abs(actual - predicted).sum()
            / np.abs(actual).sum()
        ) * 100
    }


all_predictions = []
all_yearly_metrics = []

for dataset_name, path in DATASETS.items():

    print(f"\n{'=' * 90}")
    print(dataset_name)

    df = pd.read_csv(path)

    df = df[
        (df["TID"] >= FIRST_TRAIN_YEAR) &
        (df["TID"] <= LAST_YEAR)
    ].copy()

    df = df.sort_values(["TID", "ANVEND"]).reset_index(drop=True)

    categorical = ["ANVEND"]
    numeric = [
        col for col in df.columns
        if col not in ["TID", "ANVEND", "Started"]
    ]

    features = categorical + numeric

    test_years = [
        year for year in sorted(df["TID"].unique())
        if year >= FIRST_TEST_YEAR
    ]

    print("Shape:", df.shape)
    print("Training starts:", FIRST_TRAIN_YEAR)
    print("Testing:", FIRST_TEST_YEAR, "-", LAST_YEAR)
    print("Categories:", df["ANVEND"].nunique())
    print("Features:", features)

    for test_year in test_years:

        train = df[df["TID"] < test_year].copy()
        test = df[df["TID"] == test_year].copy()

        if train.empty or test.empty:
            continue

        X_train = train[features].copy()
        y_train = train["Started"].copy()
        X_test = test[features].copy()

        # Persistence baseline
        persistence = test[
            ["TID", "ANVEND", "Started", "Started_lag1"]
        ].copy()

        persistence["Prediction"] = (
            persistence["Started_lag1"].clip(lower=0)
        )
        persistence["Model"] = "Persistence"
        persistence["Dataset"] = dataset_name

        all_predictions.append(
            persistence.rename(columns={"Started": "Actual"})
        )

        # sklearn-compatible models
        models = build_models()

        for model_name, estimator in models.items():

            preprocessor = build_preprocessor(
                categorical,
                numeric,
                scale_numeric=(model_name == "Ridge")
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator)
            ])

            pipeline.fit(X_train, y_train)

            predicted = np.maximum(
                pipeline.predict(X_test),
                0
            )

            result = test[
                ["TID", "ANVEND", "Started"]
            ].copy()

            result["Prediction"] = predicted
            result["Model"] = model_name
            result["Dataset"] = dataset_name

            all_predictions.append(
                result.rename(columns={"Started": "Actual"})
            )

        # CatBoost with native categorical handling
        cat_train = X_train.copy()
        cat_test = X_test.copy()

        cat_train["ANVEND"] = cat_train["ANVEND"].astype(str)
        cat_test["ANVEND"] = cat_test["ANVEND"].astype(str)

        catboost = CatBoostRegressor(
            iterations=500,
            depth=5,
            learning_rate=0.03,
            loss_function="RMSE",
            l2_leaf_reg=5,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1
        )

        catboost.fit(
            cat_train,
            y_train,
            cat_features=["ANVEND"]
        )

        cat_predicted = np.maximum(
            catboost.predict(cat_test),
            0
        )

        result = test[
            ["TID", "ANVEND", "Started"]
        ].copy()

        result["Prediction"] = cat_predicted
        result["Model"] = "CatBoost"
        result["Dataset"] = dataset_name

        all_predictions.append(
            result.rename(columns={"Started": "Actual"})
        )

        print(
            f"Completed test year {test_year} | "
            f"Training rows: {len(train)} | "
            f"Test rows: {len(test)}"
        )


# Combine predictions
predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions.to_csv(
    OUTPUT / "common_period_rolling_predictions.csv",
    index=False
)


# Pooled metrics
summary_rows = []

for (dataset, model), group in predictions.groupby(
    ["Dataset", "Model"]
):
    metrics = calculate_metrics(
        group["Actual"],
        group["Prediction"]
    )

    summary_rows.append({
        "Dataset": dataset,
        "Model": model,
        **metrics,
        "Test_Years": group["TID"].nunique()
    })

summary = pd.DataFrame(summary_rows).sort_values(
    ["Dataset", "RMSE"]
)

summary.to_csv(
    OUTPUT / "common_period_model_comparison.csv",
    index=False
)


# Year-level metrics
yearly_rows = []

for (dataset, model, year), group in predictions.groupby(
    ["Dataset", "Model", "TID"]
):
    metrics = calculate_metrics(
        group["Actual"],
        group["Prediction"]
    )

    yearly_rows.append({
        "Dataset": dataset,
        "Model": model,
        "Year": year,
        **metrics
    })

yearly = pd.DataFrame(yearly_rows)

yearly.to_csv(
    OUTPUT / "common_period_yearly_metrics.csv",
    index=False
)


# Mean annual performance
annual_summary = (
    yearly.groupby(["Dataset", "Model"])[
        ["MAE", "RMSE", "WAPE_pct"]
    ]
    .mean()
    .reset_index()
    .sort_values(["Dataset", "RMSE"])
)

annual_summary.to_csv(
    OUTPUT / "common_period_mean_yearly_metrics.csv",
    index=False
)


# Direct comparison of macro feature contribution
wide = summary.pivot(
    index="Model",
    columns="Dataset",
    values=["MAE", "RMSE", "WAPE_pct"]
)

comparison_rows = []

for model in sorted(predictions["Model"].unique()):

    if model == "CatBoost":
        # CatBoost is not affected by the one-hot feature representation,
        # but its native categorical handling remains unchanged.
        pass

    full = summary[
        (summary["Dataset"] == "Full_History_Common") &
        (summary["Model"] == model)
    ]

    macro = summary[
        (summary["Dataset"] == "Macro_Enhanced_Common") &
        (summary["Model"] == model)
    ]

    if full.empty or macro.empty:
        continue

    full_rmse = full["RMSE"].iloc[0]
    macro_rmse = macro["RMSE"].iloc[0]

    comparison_rows.append({
        "Model": model,
        "Full_History_RMSE": full_rmse,
        "Macro_Enhanced_RMSE": macro_rmse,
        "RMSE_Improvement_pct": (
            (full_rmse - macro_rmse) / full_rmse
        ) * 100,
        "Full_History_WAPE": full["WAPE_pct"].iloc[0],
        "Macro_Enhanced_WAPE": macro["WAPE_pct"].iloc[0]
    })

macro_comparison = pd.DataFrame(
    comparison_rows
).sort_values(
    "RMSE_Improvement_pct",
    ascending=False
)

macro_comparison.to_csv(
    OUTPUT / "macro_incremental_value.csv",
    index=False
)


print("\n" + "=" * 90)
print("COMMON-PERIOD POOLED MODEL COMPARISON")
print(summary.round(4).to_string(index=False))

print("\n" + "=" * 90)
print("MEAN YEARLY PERFORMANCE")
print(annual_summary.round(4).to_string(index=False))

print("\n" + "=" * 90)
print("INCREMENTAL MACRO FEATURE VALUE")
print(macro_comparison.round(4).to_string(index=False))

print("\nOUTPUTS SAVED TO:", OUTPUT)
