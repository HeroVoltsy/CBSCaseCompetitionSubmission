
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.makedirs("data/processed/modeling", exist_ok=True)

df = pd.read_csv(
    "data/processed/market_indicators/construction_macro_panel.csv"
).sort_values("TID").copy()

df["starts_growth_pct"] = df["Started"].pct_change() * 100
df["gdp_growth_pct"] = (
    df["real_gdp_2020_chained"].pct_change() * 100
)
df["change_lending_rate"] = (
    df["corporate_lending_rate_pct"].diff()
)
df["crisis_2009"] = (df["TID"] == 2009).astype(int)

columns = [
    "TID",
    "starts_growth_pct",
    "gdp_growth_pct",
    "corporate_lending_rate_pct",
    "change_lending_rate",
    "inflation_pct",
    "crisis_2009"
]

data = (
    df[columns]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)

specifications = {
    "M1_Original": [
        "gdp_growth_pct",
        "corporate_lending_rate_pct",
        "inflation_pct"
    ],
    "M2_Change_Rate": [
        "gdp_growth_pct",
        "change_lending_rate",
        "inflation_pct"
    ],
    "M3_Crisis_Adjusted": [
        "gdp_growth_pct",
        "change_lending_rate",
        "inflation_pct",
        "crisis_2009"
    ]
}

results = []
diagnostics = []

for spec_name, predictors in specifications.items():
    X = sm.add_constant(data[predictors])
    y = data["starts_growth_pct"]

    model = sm.OLS(y, X).fit()
    hac = model.get_robustcov_results(
        cov_type="HAC",
        maxlags=1,
        use_correction=True
    )

    names = model.model.exog_names
    params = np.asarray(model.params)
    ses = np.asarray(hac.bse)
    pvalues = np.asarray(hac.pvalues)
    intervals = np.asarray(hac.conf_int())

    for i, name in enumerate(names):
        partial_r2 = np.nan
        partial_f2 = np.nan
        standardized_beta = np.nan

        if name != "const":
            standardized_beta = (
                params[i] * data[name].std() / y.std()
            )

            reduced_predictors = [
                p for p in predictors if p != name
            ]

            reduced_X = sm.add_constant(
                data[reduced_predictors]
            )

            reduced_model = sm.OLS(y, reduced_X).fit()

            partial_r2 = (
                reduced_model.ssr - model.ssr
            ) / reduced_model.ssr

            partial_f2 = partial_r2 / (1 - partial_r2)

        results.append({
            "Specification": spec_name,
            "Variable": name,
            "Coefficient": params[i],
            "Standardized_Beta": standardized_beta,
            "HAC1_SE": ses[i],
            "HAC1_p": pvalues[i],
            "HAC1_CI_Lower": intervals[i, 0],
            "HAC1_CI_Upper": intervals[i, 1],
            "Partial_R2": partial_r2,
            "Partial_Cohens_f2": partial_f2
        })

    diagnostics.append({
        "Specification": spec_name,
        "N": int(model.nobs),
        "R_squared": model.rsquared,
        "Adjusted_R_squared": model.rsquared_adj,
        "AIC": model.aic,
        "BIC": model.bic,
        "OLS_F_p": model.f_pvalue
    })

effects = pd.DataFrame(results)
fit = pd.DataFrame(diagnostics)

effects.to_csv(
    "data/processed/modeling/specification_effects.csv",
    index=False
)

fit.to_csv(
    "data/processed/modeling/specification_fit.csv",
    index=False
)

print("\n===== SPECIFICATION FIT =====")
print(fit.round(4).to_string(index=False))

print("\n===== HAC EFFECTS =====")
print(effects.round(4).to_string(index=False))
