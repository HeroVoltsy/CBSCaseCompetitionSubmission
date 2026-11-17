
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson

os.makedirs("data/processed/modeling", exist_ok=True)

df = pd.read_csv(
    "data/processed/market_indicators/construction_macro_panel.csv"
).sort_values("TID").copy()

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
    df[["TID", target] + features]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)

X = sm.add_constant(data[features])
y = data[target]

model = sm.OLS(y, X).fit()

stationarity = []

for column in [target] + features:
    series = data[column].dropna()

    adf = adfuller(series, regression="c", autolag="AIC")
    kp = kpss(series, regression="c", nlags="auto")

    stationarity.append({
        "Variable": column,
        "ADF_Statistic": adf[0],
        "ADF_p": adf[1],
        "ADF_Lags": adf[2],
        "KPSS_Statistic": kp[0],
        "KPSS_p": kp[1]
    })

stationarity = pd.DataFrame(stationarity)

stationarity.to_csv(
    "data/processed/modeling/stationarity_tests.csv",
    index=False
)

influence = model.get_influence()

cook = influence.cooks_distance[0]

sensitivity = []

for i in range(len(data)):
    reduced_data = data.drop(index=i)

    reduced_X = sm.add_constant(reduced_data[features])
    reduced_y = reduced_data[target]

    reduced_model = sm.OLS(
        reduced_y, reduced_X
    ).fit()

    sensitivity.append({
        "Excluded_Year": int(data.loc[i, "TID"]),
        "Observations": len(reduced_data),
        "R_squared": reduced_model.rsquared,
        "Adjusted_R_squared": reduced_model.rsquared_adj,
        "GDP_Coefficient": reduced_model.params["gdp_growth_pct"],
        "GDP_p": reduced_model.pvalues["gdp_growth_pct"],
        "Lending_Coefficient": reduced_model.params["corporate_lending_rate_pct"],
        "Lending_p": reduced_model.pvalues["corporate_lending_rate_pct"],
        "Inflation_Coefficient": reduced_model.params["inflation_pct"],
        "Inflation_p": reduced_model.pvalues["inflation_pct"]
    })

sensitivity = pd.DataFrame(sensitivity)

sensitivity.to_csv(
    "data/processed/modeling/leave_one_out_sensitivity.csv",
    index=False
)

data["Cook_Distance"] = cook

print("\n===== STATIONARITY TESTS =====")
print(stationarity.round(4).to_string(index=False))

print("\n===== LEAVE-ONE-OUT SENSITIVITY =====")
print(sensitivity.round(4).to_string(index=False))

print("\n===== MOST INFLUENTIAL OBSERVATIONS =====")
print(
    data[["TID", "Cook_Distance"]]
    .sort_values("Cook_Distance", ascending=False)
    .head(5)
    .to_string(index=False)
)

print("\n===== FULL-SAMPLE COEFFICIENTS =====")
print(model.params.round(4))

print("\n===== FULL-SAMPLE R-SQUARED =====")
print(round(model.rsquared, 4))

print("\n===== DURBIN-WATSON =====")
print(round(durbin_watson(model.resid), 4))
