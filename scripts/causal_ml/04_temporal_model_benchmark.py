
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

base = Path("data/processed/causal_ml")
output = base
output.mkdir(parents=True, exist_ok=True)

datasets = {
    "Full_History": {
        "path": base / "construction_panel_modeling.csv",
        "first_test_year": 1999
    },
    "Macro_Enhanced": {
        "path": base / "construction_panel_macro_modeling.csv",
        "first_test_year": 2013
    }
}

results = []
yearly_results = []

for dataset_name, config in datasets.items():
    df = pd.read_csv(config["path"])

    categorical = ["ANVEND"]
    numeric = [c for c in df.columns if c not in ["TID", "ANVEND", "Started"]]
    features = categorical + numeric

    years = sorted(df["TID"].unique())
    test_years = [y for y in years if y >= config["first_test_year"]]

    predictions = []

    for test_year in test_years:
        train = df[df["TID"] < test_year].copy()
        test = df[df["TID"] == test_year].copy()

        if train.empty or test.empty:
            continue

        X_train = train[features].copy()
        y_train = train["Started"]
        X_test = test[features].copy()

        baseline = test[
            ["TID", "ANVEND", "Started", "Started_lag1"]
        ].copy()
        baseline["Prediction"] = baseline["Started_lag1"].clip(lower=0)
        baseline["Model"] = "Persistence"
        baseline["Dataset"] = dataset_name
        predictions.append(
            baseline.rename(columns={"Started": "Actual"})
        )

        estimators = {
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

        for model_name, estimator in estimators.items():
            if model_name == "Ridge":
                preprocessor = ColumnTransformer([
                    (
                        "category",
                        OneHotEncoder(handle_unknown="ignore"),
                        categorical
                    ),
                    (
                        "numeric",
                        Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler())
                        ]),
                        numeric
                    )
                ])
            else:
                preprocessor = ColumnTransformer([
                    (
                        "category",
                        OneHotEncoder(handle_unknown="ignore"),
                        categorical
                    ),
                    (
                        "numeric",
                        SimpleImputer(strategy="median"),
                        numeric
                    )
                ])

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator)
            ])

            pipeline.fit(X_train, y_train)
            pred = np.maximum(pipeline.predict(X_test), 0)

            result = test[["TID", "ANVEND", "Started"]].copy()
            result["Prediction"] = pred
            result["Model"] = model_name
            result["Dataset"] = dataset_name
            predictions.append(
                result.rename(columns={"Started": "Actual"})
            )

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

        cat_pred = np.maximum(catboost.predict(cat_test), 0)

        result = test[["TID", "ANVEND", "Started"]].copy()
        result["Prediction"] = cat_pred
        result["Model"] = "CatBoost"
        result["Dataset"] = dataset_name
        predictions.append(
            result.rename(columns={"Started": "Actual"})
        )

    pred_df = pd.concat(predictions, ignore_index=True)

    pred_df.to_csv(
        output / f"{dataset_name.lower()}_rolling_predictions.csv",
        index=False
    )

    for model_name, group in pred_df.groupby("Model"):
        actual = group["Actual"].to_numpy()
        predicted = group["Prediction"].to_numpy()

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "N": len(group),
            "Test_Years": group["TID"].nunique(),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted),
            "WAPE_pct": (
                np.abs(actual - predicted).sum()
                / np.abs(actual).sum()
            ) * 100
        })

        for year, yearly in group.groupby("TID"):
            yearly_results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Year": year,
                "N": len(yearly),
                "MAE": mean_absolute_error(
                    yearly["Actual"], yearly["Prediction"]
                ),
                "RMSE": np.sqrt(mean_squared_error(
                    yearly["Actual"], yearly["Prediction"]
                )),
                "WAPE_pct": (
                    np.abs(
                        yearly["Actual"] - yearly["Prediction"]
                    ).sum()
                    / np.abs(yearly["Actual"]).sum()
                ) * 100
            })

summary = pd.DataFrame(results).sort_values(
    ["Dataset", "RMSE"]
)
yearly_summary = pd.DataFrame(yearly_results)

summary.to_csv(
    output / "temporal_model_comparison.csv",
    index=False
)
yearly_summary.to_csv(
    output / "temporal_yearly_metrics.csv",
    index=False
)

print("\nTEMPORAL MODEL COMPARISON")
print(summary.round(4).to_string(index=False))

print("\nMEAN YEARLY PERFORMANCE")
print(
    yearly_summary.groupby(["Dataset", "Model"])[
        ["MAE", "RMSE", "WAPE_pct"]
    ]
    .mean()
    .round(4)
    .to_string()
)

