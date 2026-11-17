from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FULL_PATH = (
    ROOT / "data/processed/causal_ml/"
    "construction_panel_modeling.csv"
)
MACRO_PATH = (
    ROOT / "data/processed/causal_ml/"
    "construction_panel_macro_modeling.csv"
)

OUT = (
    ROOT / "data/processed/causal_ml/"
    "explainability"
)
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "Started"
YEAR = "TID"
CATEGORY = "ANVEND"

CONSTRUCTION_FEATURES = [
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

N_GROUPS = 6
N_TEST_GROUPS = 2
EMBARGO_YEARS = 1

N_PERMUTATIONS = 20
MAX_SHAP_ROWS_PER_SPLIT = 100
RANDOM_STATE = 42

DATASETS = {
    "Full_History": {
        "path": FULL_PATH,
        "features": CONSTRUCTION_FEATURES,
    },
    "Macro_Enhanced": {
        "path": MACRO_PATH,
        "features": CONSTRUCTION_FEATURES + MACRO_FEATURES,
    },
}

# Persistence is a benchmark, not a fitted model to explain.
MODEL_NAMES = {
    "Full_History": ["Ridge", "RandomForest", "XGBoost"],
    "Macro_Enhanced": ["RandomForest", "XGBoost"],
}


# ============================================================
# PREPROCESSING AND MODELS
# ============================================================

def build_pipeline(features, model_name):
    numeric_features = list(features)
    categorical_features = [CATEGORY]

    scale_numeric = model_name == "Ridge"

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

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

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    if model_name == "Ridge":
        model = Ridge(alpha=10.0)

    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])


# ============================================================
# CPCV SPLITS
# ============================================================

def construct_splits(df):
    years = np.sort(df[YEAR].unique())

    year_groups = np.array_split(years, N_GROUPS)

    group_lookup = {}
    for group_id, group_years in enumerate(year_groups):
        for year in group_years:
            group_lookup[int(year)] = group_id

    splits = []

    for split_number, test_group_ids in enumerate(
        itertools.combinations(
            range(N_GROUPS),
            N_TEST_GROUPS,
        )
    ):
        test_years = sorted(
            int(year)
            for group_id in test_group_ids
            for year in year_groups[group_id]
        )

        embargo_years = set()
        for year in test_years:
            for offset in range(
                -EMBARGO_YEARS,
                EMBARGO_YEARS + 1,
            ):
                embargo_years.add(year + offset)

        test_mask = df[YEAR].isin(test_years)
        train_mask = ~df[YEAR].isin(embargo_years)

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        splits.append({
            "split": split_number,
            "test_groups": tuple(test_group_ids),
            "test_years": test_years,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "group_lookup": group_lookup,
        })

    return splits


# ============================================================
# TEMPORALLY STRUCTURED PERMUTATION
# ============================================================

def circular_year_permutation(
    frame,
    columns,
    test_years,
    group_lookup,
    rng,
):
    """
    Circularly shift complete year blocks within each CPCV
    temporal group. All categories from a year move together.
    """

    permuted = frame.copy()

    for group_id in sorted({
        group_lookup[int(year)]
        for year in test_years
    }):
        group_test_years = sorted(
            year
            for year in test_years
            if group_lookup[int(year)] == group_id
        )

        if len(group_test_years) < 2:
            continue

        shift = int(rng.integers(1, len(group_test_years)))

        source_years = (
            group_test_years[-shift:]
            + group_test_years[:-shift]
        )

        year_mapping = dict(
            zip(group_test_years, source_years)
        )

        for target_year, source_year in year_mapping.items():
            target_mask = (
                permuted[YEAR].to_numpy() == target_year
            )
            source_mask = (
                frame[YEAR].to_numpy() == source_year
            )

            target_positions = np.flatnonzero(target_mask)
            source_positions = np.flatnonzero(source_mask)

            target_block = permuted.iloc[
                target_positions
            ].copy()

            source_block = frame.iloc[
                source_positions
            ].copy()

            target_block = target_block.sort_values(
                CATEGORY
            )
            source_block = source_block.sort_values(
                CATEGORY
            )

            target_positions_sorted = (
                permuted.iloc[target_positions]
                .sort_values(CATEGORY)
                .index
            )

            source_values = (
                source_block[columns]
                .to_numpy()
            )

            permuted.loc[
                target_positions_sorted,
                columns,
            ] = source_values

    return permuted


def permute_category_within_year(frame, rng):
    permuted = frame.copy()

    for year in frame[YEAR].unique():
        mask = frame[YEAR] == year
        positions = np.flatnonzero(mask.to_numpy())

        values = frame.iloc[positions][CATEGORY].to_numpy()
        shuffled = rng.permutation(values)

        permuted.iloc[
            positions,
            permuted.columns.get_loc(CATEGORY),
        ] = shuffled

    return permuted


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def get_permutation_groups(features):
    groups = {
        feature: [feature]
        for feature in features
    }

    groups["Construction_lag_block"] = (
        CONSTRUCTION_FEATURES.copy()
    )

    if any(feature in MACRO_FEATURES for feature in features):
        groups["Macro_feature_block"] = [
            feature
            for feature in MACRO_FEATURES
            if feature in features
        ]

    groups["Building_use_category"] = [CATEGORY]

    return groups


def calculate_permutation_importance(
    pipeline,
    X_test,
    y_test,
    split,
    dataset_name,
    model_name,
    features,
):
    rng = np.random.default_rng(
        RANDOM_STATE + split["split"]
    )

    baseline_prediction = pipeline.predict(X_test)
    baseline_mae = mean_absolute_error(
        y_test,
        baseline_prediction,
    )

    importance_groups = get_permutation_groups(features)
    rows = []

    for group_name, columns in importance_groups.items():
        if not columns:
            continue

        for repetition in range(N_PERMUTATIONS):
            if group_name == "Building_use_category":
                X_permuted = permute_category_within_year(
                    X_test,
                    rng,
                )
            else:
                X_permuted = circular_year_permutation(
                    X_test,
                    columns,
                    split["test_years"],
                    split["group_lookup"],
                    rng,
                )

            permuted_prediction = pipeline.predict(
                X_permuted
            )

            permuted_mae = mean_absolute_error(
                y_test,
                permuted_prediction,
            )

            rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Split": split["split"],
                "Test_Groups": str(split["test_groups"]),
                "Feature": group_name,
                "Permutation": repetition,
                "Baseline_MAE": baseline_mae,
                "Permuted_MAE": permuted_mae,
                "Importance_MAE": (
                    permuted_mae - baseline_mae
                ),
                "N_Test": len(y_test),
            })

    return rows


# ============================================================
# SHAP
# ============================================================

def shap_feature_name_mapping(transformed_names):
    mapped = []

    for name in transformed_names:
        if "__" in name:
            name = name.split("__", 1)[1]

        if name.startswith("ANVEND_"):
            name = CATEGORY

        mapped.append(name)

    return mapped


def calculate_shap(
    pipeline,
    X_test,
    split,
    dataset_name,
    model_name,
):
    if not SHAP_AVAILABLE:
        return [], [], None

    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]

    rng = np.random.default_rng(
        RANDOM_STATE + split["split"] + 1000
    )

    if len(X_test) > MAX_SHAP_ROWS_PER_SPLIT:
        sample_idx = rng.choice(
            len(X_test),
            size=MAX_SHAP_ROWS_PER_SPLIT,
            replace=False,
        )
        X_sample = X_test.iloc[sample_idx].copy()
    else:
        X_sample = X_test.copy()

    try:
        X_transformed = prep.transform(X_sample)

        feature_names = prep.get_feature_names_out()
        mapped_names = shap_feature_name_mapping(
            feature_names
        )

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(
            X_transformed
        )

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 0]

        if shap_values.shape[1] != len(mapped_names):
            raise ValueError(
                "SHAP feature dimensions do not match "
                "transformed feature names."
            )

        mean_abs = np.mean(
            np.abs(shap_values),
            axis=0,
        )

        shap_rows = []

        for feature_name, importance in zip(
            mapped_names,
            mean_abs,
        ):
            shap_rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Split": split["split"],
                "Feature": feature_name,
                "Mean_Abs_SHAP": float(importance),
                "N_Explained": len(X_sample),
            })

        local_rows = []

        transformed_df = pd.DataFrame(
            X_transformed,
            columns=mapped_names,
        )

        for row_idx in range(len(X_sample)):
            for feature_idx, feature_name in enumerate(
                mapped_names
            ):
                local_rows.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Split": split["split"],
                    "Feature": feature_name,
                    "SHAP_Value": float(
                        shap_values[row_idx, feature_idx]
                    ),
                    "Feature_Value": float(
                        transformed_df.iloc[
                            row_idx,
                            feature_idx,
                        ]
                    ),
                })

        return shap_rows, local_rows, None

    except Exception as exc:
        return [], [], str(exc)


# ============================================================
# MAIN
# ============================================================

def main():
    permutation_rows = []
    shap_rows = []
    local_shap_rows = []
    shap_errors = []

    for dataset_name, config in DATASETS.items():
        path = config["path"]
        features = config["features"]

        print(f"\n===== {dataset_name} =====")

        df = pd.read_csv(path)

        required = (
            [YEAR, CATEGORY, TARGET]
            + features
        )

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {path.name}: {missing}"
            )

        df = df[required].copy()
        df[YEAR] = pd.to_numeric(
            df[YEAR],
            errors="coerce",
        )

        df[TARGET] = pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )

        df = df.dropna(
            subset=[YEAR, CATEGORY, TARGET]
        ).reset_index(drop=True)

        df[YEAR] = df[YEAR].astype(int)

        splits = construct_splits(df)

        print(
            f"Rows: {len(df):,} | "
            f"Years: {df[YEAR].nunique()} | "
            f"CPCV splits: {len(splits)}"
        )

        for model_name in MODEL_NAMES[dataset_name]:
            print(f"\n--- {model_name} ---")

            for split in splits:
                train = df.iloc[
                    split["train_idx"]
                ].copy()

                test = df.iloc[
                    split["test_idx"]
                ].copy()

                X_train = train[
                    features + [CATEGORY]
                ]
                y_train = train[TARGET]

                X_test = test[
                    features + [CATEGORY, YEAR]
                ]
                y_test = test[TARGET]

                pipeline = build_pipeline(
                    features,
                    model_name,
                )

                pipeline.fit(X_train, y_train)

                permutation_rows.extend(
                    calculate_permutation_importance(
                        pipeline,
                        X_test,
                        y_test,
                        split,
                        dataset_name,
                        model_name,
                        features,
                    )
                )

                if model_name in [
                    "RandomForest",
                    "XGBoost",
                ]:
                    current_shap, current_local, error = (
                        calculate_shap(
                            pipeline,
                            X_test,
                            split,
                            dataset_name,
                            model_name,
                        )
                    )

                    shap_rows.extend(current_shap)
                    local_shap_rows.extend(current_local)

                    if error:
                        shap_errors.append({
                            "Dataset": dataset_name,
                            "Model": model_name,
                            "Split": split["split"],
                            "Error": error,
                        })

                print(
                    f"Split {split['split'] + 1:02d}/"
                    f"{len(splits):02d} | "
                    f"Train={len(train):4d} | "
                    f"Test={len(test):4d}"
                )

    # ========================================================
    # SAVE PERMUTATION RESULTS
    # ========================================================

    permutation_df = pd.DataFrame(permutation_rows)

    permutation_df.to_csv(
        OUT / "permutation_importance_by_split.csv",
        index=False,
    )

    permutation_summary = (
        permutation_df
        .groupby(
            ["Dataset", "Model", "Feature"],
            as_index=False,
        )
        .agg(
            Mean_Importance=("Importance_MAE", "mean"),
            SD_Importance=("Importance_MAE", "std"),
            Median_Importance=("Importance_MAE", "median"),
            Positive_Share=(
                "Importance_MAE",
                lambda x: (x > 0).mean(),
            ),
            Mean_Baseline_MAE=("Baseline_MAE", "mean"),
            N_Permutations=("Importance_MAE", "count"),
        )
    )

    permutation_summary["Importance_Rank"] = (
        permutation_summary
        .groupby(["Dataset", "Model"])[
            "Mean_Importance"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    permutation_summary.to_csv(
        OUT / "permutation_importance_summary.csv",
        index=False,
    )

    # ========================================================
    # SAVE SHAP RESULTS
    # ========================================================

    if shap_rows:
        shap_df = pd.DataFrame(shap_rows)

        shap_df.to_csv(
            OUT / "shap_importance_by_split.csv",
            index=False,
        )

        shap_summary = (
            shap_df
            .groupby(
                ["Dataset", "Model", "Feature"],
                as_index=False,
            )
            .agg(
                Mean_Abs_SHAP=("Mean_Abs_SHAP", "mean"),
                SD_Abs_SHAP=("Mean_Abs_SHAP", "std"),
                Median_Abs_SHAP=("Mean_Abs_SHAP", "median"),
                Positive_Split_Share=(
                    "Mean_Abs_SHAP",
                    lambda x: (x > 0).mean(),
                ),
                N_Splits=("Split", "nunique"),
            )
        )

        shap_summary["Importance_Rank"] = (
            shap_summary
            .groupby(["Dataset", "Model"])[
                "Mean_Abs_SHAP"
            ]
            .rank(
                ascending=False,
                method="min",
            )
        )

        shap_summary.to_csv(
            OUT / "shap_importance_summary.csv",
            index=False,
        )

    if local_shap_rows:
        pd.DataFrame(local_shap_rows).to_csv(
            OUT / "shap_local_values.csv",
            index=False,
        )

    if shap_errors:
        pd.DataFrame(shap_errors).to_csv(
            OUT / "shap_errors.csv",
            index=False,
        )

    # ========================================================
    # PLOTS
    # ========================================================

    for (dataset, model), group in (
        permutation_summary.groupby(
            ["Dataset", "Model"]
        )
    ):
        plot_data = (
            group
            .sort_values("Mean_Importance")
            .tail(15)
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(
            plot_data["Feature"],
            plot_data["Mean_Importance"],
        )

        ax.axvline(
            0,
            linewidth=1,
            linestyle="--",
        )

        ax.set_title(
            f"Temporal CPCV Permutation Importance\n"
            f"{dataset} | {model}"
        )
        ax.set_xlabel(
            "Increase in held-out MAE after permutation"
        )
        ax.set_ylabel("Feature")

        fig.tight_layout()

        fig.savefig(
            OUT / (
                f"permutation_{dataset}_{model}.png"
            ),
            dpi=200,
        )
        plt.close(fig)

    if shap_rows:
        for (dataset, model), group in (
            shap_summary.groupby(
                ["Dataset", "Model"]
            )
        ):
            plot_data = (
                group
                .sort_values("Mean_Abs_SHAP")
                .tail(15)
            )

            fig, ax = plt.subplots(figsize=(10, 7))

            ax.barh(
                plot_data["Feature"],
                plot_data["Mean_Abs_SHAP"],
            )

            ax.set_title(
                f"Temporal CPCV Mean Absolute SHAP\n"
                f"{dataset} | {model}"
            )
            ax.set_xlabel("Mean absolute SHAP value")
            ax.set_ylabel("Feature")

            fig.tight_layout()

            fig.savefig(
                OUT / f"shap_{dataset}_{model}.png",
                dpi=200,
            )
            plt.close(fig)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n===== PERMUTATION IMPORTANCE =====")
    print(
        permutation_summary
        .sort_values(
            ["Dataset", "Model", "Importance_Rank"]
        )
        .groupby(["Dataset", "Model"])
        .head(10)
        .to_string(index=False)
    )

    if shap_rows:
        print("\n===== SHAP IMPORTANCE =====")
        print(
            shap_summary
            .sort_values(
                ["Dataset", "Model", "Importance_Rank"]
            )
            .groupby(["Dataset", "Model"])
            .head(10)
            .to_string(index=False)
        )

    print(f"\nSaved to: {OUT}")

    if not SHAP_AVAILABLE:
        print(
            "\nSHAP unavailable. Install with: "
            "python3 -m pip install shap"
        )

    if shap_errors:
        print(
            f"\nSHAP errors in {len(shap_errors)} "
            "split/model combinations. See shap_errors.csv."
        )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )
    main()
