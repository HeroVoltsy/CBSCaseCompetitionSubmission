
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

p = Path("data/processed/market_indicators/construction_macro_panel.csv")
out = Path("data/processed/modeling")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(p).sort_values("TID")

df["gdp_growth_pct"] = df["real_gdp_2020_chained"].pct_change() * 100
df["starts_growth_pct"] = df["Started"].pct_change() * 100
df["construction_starts_m2"] = df["Started"] / 1_000_000

df["cpi_inflation_pct"] = df["cpi"].pct_change() * 100

model_df = df[
    [
        "TID",
        "starts_growth_pct",
        "gdp_growth_pct",
        "corporate_lending_rate_pct",
        "inflation_pct",
    ]
].dropna()

X = model_df[
    ["gdp_growth_pct", "corporate_lending_rate_pct", "inflation_pct"]
]
X = sm.add_constant(X)
y = model_df["starts_growth_pct"]

model = sm.OLS(y, X).fit(cov_type="HC3")

print(model.summary())

model_df["predicted_starts_growth_pct"] = model.predict(X)
model_df["residual"] = y - model_df["predicted_starts_growth_pct"]

model_df.to_csv(out / "econometric_baseline_results.csv", index=False)

with open(out / "econometric_baseline_summary.txt", "w") as f:
    f.write(model.summary().as_text())

print("\nObservations:", len(model_df))
print("R-squared:", round(model.rsquared, 4))
print("Adjusted R-squared:", round(model.rsquared_adj, 4))
print("Saved results:", out)
