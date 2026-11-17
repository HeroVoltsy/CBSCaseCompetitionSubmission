
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import pearsonr, jarque_bera
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    acorr_breusch_godfrey
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

os.makedirs("data/processed/modeling", exist_ok=True)

df = pd.read_csv(
    "data/processed/market_indicators/construction_macro_panel.csv"
)

df = df.sort_values("TID").copy()

df["starts_growth_pct"] = df["Started"].pct_change() * 100
df["gdp_growth_pct"] = (
    df["real_gdp_2020_chained"].pct_change() * 100
)

features = [
    "gdp_growth_pct",
    "corporate_lending_rate_pct",
    "inflation_pct"
]

target = "starts_growth_pct"

data = (
    df[[target] + features]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

X = sm.add_constant(data[features])
y = data[target]

model = sm.OLS(y, X).fit()

hc3 = model.get_robustcov_results(cov_type="HC3")

hac1 = model.get_robustcov_results(
    cov_type="HAC",
    maxlags=1,
    use_correction=True
)

hac2 = model.get_robustcov_results(
    cov_type="HAC",
    maxlags=2,
    use_correction=True
)

names = model.model.exog_names
effect_rows = []

for i, name in enumerate(names):
    beta = float(model.params.iloc[i])

    sd_ratio = (
        data[name].std() / y.std()
        if name != "const"
        else np.nan
    )

    partial_r2 = np.nan
    partial_f2 = np.nan

    if name != "const":
        reduced_features = [
            feature for feature in features
            if feature != name
        ]

        reduced_X = sm.add_constant(data[reduced_features])
        reduced_model = sm.OLS(y, reduced_X).fit()

        partial_r2 = (
            reduced_model.ssr - model.ssr
        ) / reduced_model.ssr

        partial_f2 = partial_r2 / (1 - partial_r2)

    row = {
        "Variable": name,
        "Coefficient": beta,
        "Standardized_Beta": beta * sd_ratio,
        "Partial_R2": partial_r2,
        "Partial_Cohens_f2": partial_f2
    }

    for label, result in [
        ("OLS", model),
        ("HC3", hc3),
        ("HAC1", hac1),
        ("HAC2", hac2)
    ]:
        ci = np.asarray(result.conf_int())
        se = np.asarray(result.bse)
        t = np.asarray(result.tvalues)
        p = np.asarray(result.pvalues)

        row[f"{label}_SE"] = float(se[i])
        row[f"{label}_t"] = float(t[i])
        row[f"{label}_p"] = float(p[i])
        row[f"{label}_CI_Lower"] = float(ci[i, 0])
        row[f"{label}_CI_Upper"] = float(ci[i, 1])

    effect_rows.append(row)

effects = pd.DataFrame(effect_rows)

effects.to_csv(
    "data/processed/modeling/effect_sizes_robust_inference.csv",
    index=False
)

overall_f2 = model.rsquared / (1 - model.rsquared)

correlations = []

for feature in features:
    r, p = pearsonr(data[feature], y)

    correlations.append({
        "Variable": feature,
        "Pearson_r": r,
        "Pearson_p": p,
        "Pearson_r_squared": r ** 2
    })

pd.DataFrame(correlations).to_csv(
    "data/processed/modeling/predictor_correlations.csv",
    index=False
)

X_vif = sm.add_constant(data[features])

vif = pd.DataFrame({
    "Variable": features,
    "VIF": [
        variance_inflation_factor(
            X_vif.values, i + 1
        )
        for i in range(len(features))
    ]
})

vif.to_csv(
    "data/processed/modeling/vif.csv",
    index=False
)

bp = het_breuschpagan(model.resid, X)
bg1 = acorr_breusch_godfrey(model, nlags=1)
bg2 = acorr_breusch_godfrey(model, nlags=2)
jb = jarque_bera(model.resid)

influence = model.get_influence()

influence_df = pd.DataFrame({
    "Year": df.loc[data.index, "TID"].values,
    "Residual": model.resid.values,
    "Standardized_Residual": influence.resid_studentized_internal,
    "Cook_Distance": influence.cooks_distance[0],
    "Leverage": influence.hat_matrix_diag
})

influence_df.to_csv(
    "data/processed/modeling/influence_diagnostics.csv",
    index=False
)

diagnostics = pd.DataFrame([
    ["Observations", len(data)],
    ["R_squared", model.rsquared],
    ["Adjusted_R_squared", model.rsquared_adj],
    ["Overall_Cohens_f2", overall_f2],
    ["OLS_F_statistic", model.fvalue],
    ["OLS_F_pvalue", model.f_pvalue],
    ["Durbin_Watson", durbin_watson(model.resid)],
    ["Breusch_Pagan_LM", bp[0]],
    ["Breusch_Pagan_p", bp[1]],
    ["Breusch_Godfrey_1_p", bg1[1]],
    ["Breusch_Godfrey_2_p", bg2[1]],
    ["Jarque_Bera", jb.statistic],
    ["Jarque_Bera_p", jb.pvalue]
], columns=["Metric", "Value"])

diagnostics.to_csv(
    "data/processed/modeling/model_diagnostics.csv",
    index=False
)

print("\n===== SAMPLE =====")
print(f"Observations: {len(data)}")
print(
    f"Period: {df.loc[data.index, 'TID'].min()}–"
    f"{df.loc[data.index, 'TID'].max()}"
)

print("\n===== MODEL FIT =====")
print(model.summary())

print("\n===== EFFECT SIZES AND ROBUST INFERENCE =====")
print(effects.round(4).to_string(index=False))

print("\n===== DIAGNOSTICS =====")
print(diagnostics.round(4).to_string(index=False))

print("\n===== VIF =====")
print(vif.round(4).to_string(index=False))

print("\n===== PEARSON CORRELATIONS =====")
print(
    pd.DataFrame(correlations)
    .round(4)
    .to_string(index=False)
)

print("\n===== INFLUENTIAL OBSERVATIONS =====")
print(
    influence_df.sort_values(
        "Cook_Distance", ascending=False
    ).head(5).round(4).to_string(index=False)
)

print("\nSaved outputs to data/processed/modeling/")
