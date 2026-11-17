# CBS GLOBAL 2026 — ROCKWOOL Denmark

**Market Intelligence · Econometrics · Financial Modelling · Machine Learning · Causal Analysis**

## 1. Project Overview

This repository contains the data engineering, market research, financial modelling, econometric analysis, and machine learning workflows developed around the ROCKWOOL Denmark case for CBS GLOBAL 2026.

The central business question is:

> How can ROCKWOOL achieve 20% revenue growth in the Danish Glasswool and Building Insulation (GBI) market by 2030, and sustain growth beyond the investment period?

The project combines Danish construction-market statistics, macroeconomic indicators, company financial disclosures, scenario-based financial modelling, predictive benchmarking, and double/debiased machine learning (DML).

The analysis is designed to support a commercially actionable strategy while distinguishing observed market evidence from statistical associations, predictive results, causal assumptions, and financial projections.

**Case context:** ROCKWOOL is the case company for CBS GLOBAL 2026. Official competition information is available at [CBS Case Competition — GLOBAL](https://www.casecompetition.com/global/).

### Research principle

The objective is not to maximize the number of models or statistical results. It is to determine which commercial decisions are supported by the available evidence, quantify the financial implications of those decisions, and explicitly identify where additional company-level data are necessary.

---

## 2. Strategic Research Hypothesis

### 2.1 Central commercial hypothesis

> ROCKWOOL's Danish GBI growth constraint is not simply insufficient market demand; it is the failure to consistently convert technical preference into commercially secured purchases across the Danish building-material value chain.

The proposed strategic direction is:

**Win the Danish renovation journey by securing the conversion from specification to purchase.**

The commercial framework considers three conversion stages:

1. **Specification capture:** Getting ROCKWOOL solutions specified in eligible building projects.
2. **Specification retention:** Preventing substitution, value engineering, and material changes between specification and procurement.
3. **Purchase conversion:** Converting specified and retained projects into actual purchases through distributor coordination, product availability, stock management, and delivery.

### 2.2 Commercial causal hypothesis

The underlying commercial causal hypothesis is:

> Earlier and more consistent technical engagement, specification support, and distributor coordination increase the probability that ROCKWOOL products specified in eligible Danish GBI projects are retained through procurement and ultimately converted into purchases.

The hypothesized causal pathway is:

```text
Early Technical Engagement
          |
          v
Specification Capture
          |
          v
Specification Retention
          |
          v
Distributor Coordination
          |
          v
Purchase Conversion
          |
          v
Incremental GBI Revenue
```

The hypothesis implies that commercial interventions can improve revenue not only by generating additional demand, but also by reducing conversion losses throughout the building-material value chain.

The expected mechanisms include:

- Earlier involvement in project design and material selection.
- Improved technical specification quality.
- Reduced substitution during value engineering.
- Better coordination between specification teams and distribution partners.
- Improved product availability and procurement execution.
- Higher conversion of specified projects into realized purchases.

**Important:** This is a strategic causal hypothesis, not an empirically established result. The available aggregate construction datasets do not contain ROCKWOOL project-level intervention assignments, specifications, CRM activity, or purchase outcomes.

### 2.3 Market-level causal hypothesis

The separate statistical hypothesis investigated using DML is:

> Conditional on prior construction activity, building-use category, and deterministic time trends, an increase in building permits in the preceding year increases subsequent construction starts.

This is a market-level construction pipeline hypothesis. It is not a direct test of ROCKWOOL's commercial intervention hypothesis.

---

## 3. Case Scope and Financial Parameters

The analysis follows the case's defined Danish GBI market scope.

| Parameter | Case assumption |
|---|---:|
| Geographic market | Denmark |
| Product scope | Existing ROCKWOOL GBI products |
| Baseline annual revenue | €60 million |
| 2030 revenue target | €72 million |
| Required revenue growth | 20% |
| Incremental contribution margin | 25% |
| Annual initiative budget | €135,000 |
| Investment period | 2027–2030 |
| Total initiative budget | €540,000 |
| Distribution revenue channel | 80% |
| Direct revenue channel | 20% |
| Salesforce allocation to distribution | 70% |
| Salesforce allocation to specification | 30% |

The revenue channel split and salesforce allocation are separate assumptions and must not be conflated.

### Market scope

The project considers:

- New construction and renovation.
- Residential and non-residential buildings.
- Walls, ground, floors, and sloped roofs.

Flat roofs are outside the defined application scope.

### Financial assumption policy

The €60 million revenue baseline, €72 million target, contribution margin, and initiative budget are illustrative case parameters. They are not representations of actual ROCKWOOL Danish GBI segment disclosures.

---

## 4. Repository Structure

The repository separates original source material, processed datasets, analytical scripts, generated outputs, and research reports.

```text
CBSCaseFiles/
│
├── data/
│   ├── external_evidence/
│   ├── raw/
│   │   ├── financials/
│   │   ├── macroeconomic/
│   │   ├── metadata/
│   │   └── statbank/
│   │
│   └── processed/
│       ├── market_indicators/
│       ├── macroeconomic/
│       ├── modeling/
│       ├── financial_model/
│       │   ├── funnel_sensitivity/
│       │   └── reconciliation/
│       ├── financials/
│       │   └── extracted/
│       └── causal_ml/
│           ├── cpcv/
│           │   └── inference/
│           ├── quantile_regression/
│           ├── explainability/
│           └── dml/
│               ├── strict_pre_treatment/
│               │   └── inference/
│               └── heterogeneous_effects/
│
├── outputs/
│   ├── econometrics/
│   ├── figures/
│   └── tables/
│
├── reports/
│   └── causal_ml/
│
├── scripts/
│   ├── 01_inspect_statbank.py
│   ├── 02_summarize_metadata.py
│   ├── 03_download_market_data.py
│   ├── 04_market_analysis.py
│   ├── 05_prepare_macro.py
│   ├── 06_construction_indicators.py
│   ├── 07_econometric_baseline.py
│   ├── 08_econometric_effects.py
│   ├── 09_initiative_financial_model.py
│   ├── 10_econometric_robustness.py
│   ├── 11_econometric_specifications.py
│   ├── 12_revenue_conversion_bridge.py
│   ├── 13_integrated_initiative_valuation.py
│   ├── 14_financial_reconciliation.py
│   ├── 15_funnel_sensitivity.py
│   │
│   ├── financials/
│   │   ├── 01_download_reports.py
│   │   ├── 02_inspect_reports.py
│   │   ├── 03_extract_rockwool_xbrl.py
│   │   ├── 04_find_financial_sections.py
│   │   └── 05_rockwool_financial_mapping.py
│   │
│   └── causal_ml/
│       ├── 01_causal_data_audit.py
│       ├── 02_ml_structure_check.py
│       ├── 03_build_modeling_panel.py
│       ├── 04_temporal_model_benchmark.py
│       ├── 05_temporal_model_benchmark_per_year.py
│       ├── 06_temporal_model_cpcv_benchmark.py
│       ├── 07_cpcv_statistical_inference.py
│       ├── 08_quantile_regression.py
│       ├── 09_model_explainability.py
│       ├── 10_causal_dml.py
│       ├── 11_strict_pre_treatment_dml.py
│       ├── 12_dml_temporal_robustness.py
│       └── 13_dml_heterogeneous_effects.py
│
└── README.md
```

*The tree summarizes the principal directories and scripts. Nested XBRL package contents and individual generated files are omitted for readability.*

### Directory responsibilities

| Directory | Responsibility |
|---|---|
| `data/raw/` | Original downloaded data, reports, and source metadata |
| `data/external_evidence/` | External research and supporting evidence |
| `data/processed/` | Cleaned datasets and analytical outputs |
| `data/processed/market_indicators/` | Derived construction and building-stock indicators |
| `data/processed/macroeconomic/` | Prepared Danish macroeconomic data |
| `data/processed/modeling/` | Econometric estimates, diagnostics, and robustness results |
| `data/processed/financial_model/` | Revenue bridge, valuation, and sensitivity outputs |
| `data/processed/financials/` | Company benchmarks and extracted financial disclosures |
| `data/processed/causal_ml/` | Predictive modelling, CPCV, quantile regression, explainability, and DML |
| `outputs/figures/` | Market-analysis visualizations |
| `outputs/tables/` | Market-analysis tables |
| `outputs/econometrics/` | Econometric presentation outputs |
| `reports/causal_ml/` | Consolidated research reports |
| `scripts/` | Market analysis, econometrics, and financial modelling |
| `scripts/financials/` | Financial report extraction and benchmarking |
| `scripts/causal_ml/` | Predictive modelling, statistical inference, and DML |

---

## 5. Data Sources and Preparation

### 5.1 Danish construction statistics

The project uses Statistics Denmark (StatBank) datasets.

| Dataset | Analytical purpose |
|---|---|
| BYGV01 | Annual construction activity by phase, building use, and other dimensions |
| BYGV11 | Quarterly construction activity by phase and region |
| BYGB34 | Building stock by use and construction period |
| BYGB40 | Building stock and heated floor area |
| BYG43 | Residential building construction cost index |
| PRIS91 | Producer price index for renovation and maintenance |

The data are used to characterize:

- Construction starts, permits, completions, and buildings under construction.
- Building-use composition.
- Existing building-stock characteristics.
- Construction and renovation cost developments.
- Long-term construction-market cycles.

The datasets do not directly measure insulation consumption or ROCKWOOL market share.

### 5.2 Macroeconomic indicators

The macroeconomic panel incorporates:

- Real GDP.
- Consumer prices and inflation.
- Corporate lending rates.
- Household lending rates.

The primary overlapping annual sample for macroeconomic modelling covers 2002–2025.

Macroeconomic indicators are joined to construction activity at the annual level.

### 5.3 Company financial disclosures

The project extracts and compares 2025 financial information from:

- ROCKWOOL
- Kingspan
- Saint-Gobain
- Sto
- Owens Corning

ROCKWOOL's annual report is processed using its XBRL package. Other company reports are inspected and extracted from downloaded PDF or HTML documents.

The benchmark is intended to provide global company context, not to estimate Danish GBI segment profitability.

Accounting definitions, reporting scopes, and business mixes differ across companies. Consequently, EBITDA-like measures and operating margins are not necessarily directly comparable.

### 5.4 Data integrity

The construction-by-use panel is reconciled against national construction totals.

For the audited phases:

- Started construction.
- Permits.
- Completed construction.
- Under construction.

The maximum absolute reconciliation difference is zero.

The modelling panel also retains category and year identifiers for grouped temporal validation and inference.

---

## 6. Market Intelligence

The market-analysis workflow examines the structure and evolution of Danish construction activity.

### Main indicators

- Annual and quarterly construction activity.
- Construction activity by building use.
- Existing building stock by age and use.
- Heated floor area and heating characteristics.
- Construction cost indices.
- Renovation and maintenance price indices.

### Key market observations

- Construction starts peaked in 2021 at approximately 9.98 million m².
- Starts declined to approximately 5.08 million m² in 2025, around 49% below the 2021 peak.
- The existing building stock provides an important long-term context for renovation-oriented commercial strategies.
- Construction cost and renovation price indices describe cost developments, not renovation volume or insulation demand.

These indicators help characterize market conditions and potential exposure to construction cycles. They do not establish the size of ROCKWOOL's addressable market.

---

## 7. Econometric Analysis

### 7.1 Research question

The econometric work examines how macroeconomic conditions relate to Danish construction-start growth.

The primary outcome is:

`starts_growth_pct`

The baseline predictors are:

- GDP growth.
- Corporate lending rates.
- Inflation.

### 7.2 Estimation

The baseline specification uses ordinary least squares with robust inference.

The analysis includes:

- HC3 heteroskedasticity-robust standard errors.
- HAC standard errors.
- Alternative predictor specifications.
- Crisis-adjusted specifications.
- Variance inflation factors.
- Residual diagnostics.
- Influence and leverage analysis.
- Stationarity tests.
- Partial \(R^2\).
- Cohen's \(f^2\).
- Standardized coefficients.
- Confidence intervals.

### 7.3 Baseline results

The baseline model reports:

| Statistic | Result |
|---|---:|
| Observations | 24 |
| \(R^2\) | 0.512 |
| Adjusted \(R^2\) | 0.439 |
| F-statistic | 10.37 |
| Model p-value | 0.00025 |
| Durbin–Watson | 1.314 |

The estimated associations are:

- GDP growth: positive.
- Corporate lending rates: negative.
- Inflation: negative.

The GDP relationship is sensitive to the 2009 financial crisis. Alternative specifications and crisis-adjusted models are therefore reported rather than treating one regression as definitive.

### 7.4 Interpretation

The results describe macroeconomic associations with aggregate Danish construction activity.

They do not identify:

- Direct insulation demand.
- ROCKWOOL's Danish market share.
- ROCKWOOL-specific revenue elasticity.
- The causal impact of commercial interventions.

The small annual sample and potential temporal dependence require conservative interpretation.

---

## 8. Financial Modelling

### 8.1 Objective

The financial model evaluates whether the proposed commercial initiative can support the €12 million annual revenue uplift required by 2030.

The modelling work proceeds through:

1. Initial initiative valuation.
2. Revenue conversion bridge.
3. Integrated initiative valuation.
4. Financial reconciliation.
5. Break-even analysis.
6. One-way and factorial sensitivity analysis.

### 8.2 Revenue conversion framework

Revenue is decomposed into three conversion stages:

\[
R_t = O_t \times C_t \times S_t \times P_t
\]

Where:

- \(O_t\): normalized eligible opportunity pool.
- \(C_t\): specification capture rate.
- \(S_t\): specification retention rate.
- \(P_t\): purchase conversion rate.

The normalized opportunity pool is calibrated to the illustrative €60 million revenue baseline. It is a modelling assumption, not an independently verified estimate of the Danish GBI market.

### 8.3 Base-case assumptions

| Parameter | Baseline | 2030 base case |
|---|---:|---:|
| Specification capture | 80% | 85% |
| Specification retention | 80% | 85% |
| Purchase conversion | 78.125% | 83.125% |
| Annual revenue | €60.000m | €72.069m |

The base case produces:

- 2030 revenue: €72.069 million.
- Incremental annual revenue: €12.069 million.
- Revenue target gap: approximately +€0.069 million.

The model also evaluates conservative and upside scenarios.

### 8.4 Integrated initiative valuation

The valuation incorporates four annual initiative investments and the contribution from incremental revenue.

At a 10% discount rate:

| Scenario | NPV |
|---|---:|
| Conservative | €1.709m |
| Base | €5.128m |
| Upside | €7.509m |

The base-case IRR is approximately 535.3%.

The high IRR reflects the small assumed initiative budget relative to the modelled financial benefits. It should not be interpreted as proof that the revenue uplift is achievable.

### 8.5 Sensitivity analysis

The project evaluates sensitivity to:

- Specification capture.
- Specification retention.
- Purchase conversion.
- Revenue realization.
- Contribution margin.
- Initiative spending.
- Discount rates.

The factorial sensitivity analysis evaluates 125 deterministic combinations.

In the tested grid:

- 58 combinations achieve the 2030 revenue target.
- The minimum NPV is approximately -€0.471 million.
- The maximum NPV is approximately €11.240 million.

The 58 out of 125 result is a count of tested scenarios, not a probability of success.

### 8.6 Financial limitations

The financial model does not independently establish:

- Actual Danish GBI opportunity-pool size.
- Observed project-level conversion rates.
- Realized effectiveness of the commercial initiative.
- Complete working-capital requirements.
- Taxes and all additional implementation costs.
- Post-2030 terminal value.

The financial outputs should be interpreted as scenario-based decision support.

---

## 9. Predictive Machine Learning

### 9.1 Modelling objective

The predictive modelling work examines whether historical construction activity and macroeconomic information can improve estimates of Danish construction starts across building-use categories.

It is not designed to predict ROCKWOOL-specific sales.

### 9.2 Modelling panel

The building-use panel contains:

- 29 building-use categories.
- 23 annual observations per category in the macroeconomic modelling sample.
- 667 total observations.

The full-history construction panel extends further back in time.

Features include lagged construction activity:

- `Started_lag1`
- `Started_lag2`
- `Permits_lag1`
- `Completed_lag1`
- `Under construction_lag1`

The macro-enhanced panel additionally includes:

- `gdp_growth_pct_lag1`
- `corporate_lending_rate_pct_lag1`
- `household_lending_rate_pct_lag1`
- `inflation_pct_lag1`

### 9.3 Models

The benchmark includes:

- Persistence baseline.
- Ridge regression.
- Random Forest.
- ExtraTrees.
- XGBoost.
- LightGBM.
- CatBoost.

### 9.4 Evaluation metrics

The evaluation uses:

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- Weighted Absolute Percentage Error (WAPE).
- Coefficient of determination (\(R^2\)).
- Annual model performance.

The project compares full-history and macro-enhanced feature sets over common evaluation periods.

### 9.5 Temporal validation

The project uses expanding-window evaluation and Combinatorial Purged Cross-Validation-style temporal benchmarking (CPCV).

The CPCV configuration uses:

- Six contiguous temporal groups.
- Two test groups per split.
- Fifteen test-group combinations.
- A one-year embargo around test periods.

Observations from the same year remain grouped together.

**Validation limitation:** This CPCV implementation uses an embargo around test periods, but it is not a formal purge of overlapping label intervals. Because training data may contain years later than a particular test period, CPCV is not equivalent to strict forward-only deployment validation.

### 9.6 Statistical inference

The CPCV inference workflow:

1. Aggregates repeated test predictions at the year-category-model level.
2. Computes annual error metrics.
3. Compares models using paired annual observations.
4. Calculates paired effect sizes.
5. Uses circular block bootstrap confidence intervals.
6. Applies Holm correction for multiple comparisons.
7. Examines prediction and residual correlations.

The effective annual inference sample is 23 years, not 667 independent observations.

### 9.7 Main findings

The persistence baseline is difficult to outperform under CPCV.

No trained model demonstrates a statistically significant overall improvement over persistence after multiple-testing correction.

Macro-enhanced XGBoost shows some positive incremental signal relative to its full-history counterpart, but this advantage is not statistically significant after Holm correction across the evaluated metrics.

Performance varies by building-use category.

Potentially useful predictive signals appear in:

- Education and research buildings.
- Healthcare buildings.
- Factories and workshops.
- Multi-dwelling houses.

Other categories, including detached houses and non-residential farm buildings, are less predictable with the tested models.

These category-level results are descriptive. They should not be presented as statistically validated forecasting improvements without additional category-level inference.

---

## 10. Quantile Regression

### 10.1 Objective

Quantile regression examines whether conditional relationships between construction starts and their predictors differ across the outcome distribution.

The analysis estimates:

- 10th percentile.
- 50th percentile.
- 90th percentile.

### 10.2 Methodology

The workflow:

- Standardizes continuous predictors and the outcome.
- Includes building-use category indicators.
- Fits separate quantile regression specifications.
- Uses year-level bootstrap resampling to preserve shared annual shocks across categories.
- Compares estimated coefficients across quantiles.

### 10.3 Findings

Lagged permits show positive conditional associations with construction starts across the examined quantiles.

Some category-specific coefficient differences appear in unadjusted bootstrap intervals, but these do not establish robust distributional heterogeneity after considering multiple comparisons.

The results are exploratory conditional associations, not causal effects.

---

## 11. Double Machine Learning

### 11.1 Research question

The DML workstream examines the conditional relationship between lagged construction permits and current construction starts.

It also evaluates whether the relationship varies across broad building-use groups.

### 11.2 Formal causal hypothesis

The statistical causal hypothesis is:

**Research hypothesis \(H_1\):**

> An increase in building permits in year \(t-1\) causes an increase in construction starts in year \(t\), conditional on prior construction activity, building-use category, and underlying temporal trends.

**Null hypothesis \(H_0\):**

> Conditional on the specified pre-treatment controls and identifying assumptions, changes in lagged permits have no average causal effect on subsequent construction starts.

The treatment is continuous rather than binary. The target is therefore a conditional average treatment effect or average partial effect, interpreted as the expected change in construction starts associated with a one-unit increase in lagged permits, under the assumed identification framework.

The analysis uses:

| Role | Variable |
|---|---|
| Treatment \(D\) | `Permits_lag1` |
| Outcome \(Y\) | `Started` |
| Pre-treatment control | `Started_lag2` |
| Additional controls | Building-use category and deterministic year trends |

### 11.3 Identification assumptions

A causal interpretation requires the following assumptions.

**1. Temporal ordering and consistency**

The treatment precedes the outcome:

\[
D_{it} = Permits_{i,t-1}
\]

\[
Y_{it} = Started_{it}
\]

The observed outcome must correspond to the potential outcome under the observed treatment level. The treatment and outcome must also be defined consistently across building-use categories and years.

**2. Conditional exchangeability**

After conditioning on observed pre-treatment covariates \(X\), there must be no remaining unobserved confounding between lagged permits and current construction starts:

\[
Y_{it}(d) \perp D_{it} \mid X_{it}
\]

This is a strong assumption. Unobserved demand expectations, financing conditions, regulatory changes, planning decisions, and other time-varying market shocks may influence both permits and subsequent starts.

**3. Overlap and residual treatment variation**

There must be sufficient variation in permits across observations with comparable observed controls.

Formally:

\[
Var(D \mid X) > 0
\]

If treatment is almost perfectly predictable from the controls, the estimate may rely on limited residual variation and become unstable or difficult to interpret.

**4. No relevant interference**

The potential outcome for one building-use observation should not be materially affected by another observation's treatment in a way that violates the specified estimand.

This assumption is potentially restrictive because construction categories share macroeconomic conditions, contractors, financing markets, and supply chains.

**5. Appropriate nuisance estimation and orthogonalization**

The conditional treatment and outcome functions must be estimated sufficiently well for the orthogonal score to reduce sensitivity to nuisance-model estimation error.

Cross-fitting is used to reduce overfitting bias, but it does not eliminate omitted-variable bias or incorrect causal assumptions.

**6. Valid temporal and statistical inference**

The uncertainty estimates must account appropriately for dependence across observations within years.

Year-clustered inference and block bootstrap sensitivity analyses are used, but the number of annual clusters remains limited.

### 11.4 Cross-fitting

The nuisance models are cross-fitted using year-grouped folds, ensuring that observations from the same year remain together.

Nuisance estimators include:

- Ridge regression.
- ExtraTrees regression.

Inference uses year-clustered standard errors.

Year-grouped cross-fitting prevents same-year observations from being split between training and validation folds. It is not equivalent to chronological forward validation.

### 11.5 Expanded-control DML

The expanded-control specification initially includes lagged construction and macroeconomic controls.

The estimates are positive under both Ridge and ExtraTrees nuisance models.

However, contemporaneous construction variables and jointly determined variables may introduce post-treatment conditioning concerns. This motivates the stricter specification.

### 11.6 Strict pre-treatment DML

The strict specification uses:

- Treatment: `Permits_lag1`.
- Outcome: `Started`.
- Pre-treatment control: `Started_lag2`.
- Building-use category.
- Linear and quadratic year trends.

Results:

| Nuisance model | Estimate | Clustered SE | 95% CI |
|---|---:|---:|---:|
| Ridge | 0.982 | 0.134 | [0.704, 1.260] |
| ExtraTrees | 0.826 | 0.130 | [0.556, 1.095] |

Both specifications produce positive estimates.

The results are consistent in direction across the two nuisance-model approaches. However, the treatment remains highly predictable from the observed controls, and the analysis relies on only 23 annual clusters.

Statistical significance does not establish the causal assumptions necessary for a strong intervention interpretation.

### 11.7 Temporal robustness

Temporal robustness is examined using:

- Leave-one-year-out estimation.
- Circular moving-block bootstrap.
- Block lengths of two and three years.
- Multiple nuisance-model specifications.

All leave-one-year-out estimates remain positive.

The corrected moving-block bootstrap intervals are also positive across the examined block lengths.

These bootstrap intervals are conditional on the cross-fitted nuisance predictions. They do not fully incorporate uncertainty from refitting nuisance models in every bootstrap replication.

### 11.8 Heterogeneous effects

The heterogeneous-effects specification evaluates four building-use groups.

| Group | Number of categories |
|---|---:|
| Commercial/industrial | 5 |
| Public/institutional | 7 |
| Residential | 7 |
| Other | 10 |

The category mapping is explicitly documented in:

`data/processed/causal_ml/dml/heterogeneous_effects/category_group_mapping.csv`

The analysis estimates group-specific relationships and performs:

- Joint coefficient equality tests.
- Pairwise comparisons.
- Holm multiple-testing correction.
- Year-block bootstrap analysis.

### Omnibus heterogeneity results

| Model | Wald statistic | df | p-value |
|---|---:|---:|---:|
| Ridge | 5.649 | 3 | 0.1300 |
| ExtraTrees | 4.761 | 3 | 0.1901 |

Neither specification rejects equality of group-specific coefficients.

The estimates are broadly positive, with residential point estimates comparatively larger than public/institutional estimates, but the observed differences are not statistically significant after the formal tests.

### 11.9 Interpretation and causal limitations

The DML findings are consistent with a positive relationship between lagged building permits and subsequent construction starts.

However, the analysis does not conclusively establish that this relationship is causal because conditional exchangeability and the other identifying assumptions cannot be verified using the available aggregate observational data.

In particular:

- Unobserved demand expectations may influence both permits and starts.
- Construction financing and regulatory conditions may change over time.
- Permits and starts are linked stages of the same construction pipeline.
- Annual observations share common market shocks.
- The effective number of temporal clusters is small.

Most importantly, this DML analysis does **not** estimate the causal effect of ROCKWOOL's commercial interventions.

The project does not have ROCKWOOL project-level CRM data, randomized intervention data, or observed treatment assignment for activities such as early technical engagement, specification support, specification retention, distributor coordination, or purchase-conversion interventions.

The DML analysis examines aggregate construction-market relationships. The commercial causal hypothesis requires a separate project-level identification strategy.

---

## 12. Model Explainability

The explainability workflow is implemented in:

`scripts/causal_ml/09_model_explainability.py`

The intended analysis includes:

- Grouped temporal permutation importance.
- Feature importance under CPCV.
- SHAP-based interpretation for tree-based models.
- Examination of feature behaviour across temporal folds.

Outputs are intended to be stored in:

`data/processed/causal_ml/explainability/`

**Current status: Pending execution and output verification.**

No explainability findings should be treated as final until the generated artifacts, validation methodology, and results have been inspected.

---

## 13. Reproducibility and Execution

### 13.1 Environment

The project is developed in Python and executed primarily through a Linux/Ubuntu environment.

Core dependencies include:

- pandas
- NumPy
- SciPy
- scikit-learn
- statsmodels
- XGBoost
- LightGBM
- CatBoost
- Matplotlib
- SHAP

Exact package versions should be pinned in a requirements file before final reproducibility release.

### 13.2 Main execution groups

**Market data and indicators**

```bash
python3 scripts/01_inspect_statbank.py
python3 scripts/02_summarize_metadata.py
python3 scripts/03_download_market_data.py
python3 scripts/04_market_analysis.py
python3 scripts/05_prepare_macro.py
python3 scripts/06_construction_indicators.py
```

**Econometrics and financial modelling**

```bash
python3 scripts/07_econometric_baseline.py
python3 scripts/08_econometric_effects.py
python3 scripts/09_initiative_financial_model.py
python3 scripts/10_econometric_robustness.py
python3 scripts/11_econometric_specifications.py
python3 scripts/12_revenue_conversion_bridge.py
python3 scripts/13_integrated_initiative_valuation.py
python3 scripts/14_financial_reconciliation.py
python3 scripts/15_funnel_sensitivity.py
```

**Company financial analysis**

```bash
python3 scripts/financials/01_download_reports.py
python3 scripts/financials/02_inspect_reports.py
python3 scripts/financials/03_extract_rockwool_xbrl.py
python3 scripts/financials/04_find_financial_sections.py
python3 scripts/financials/05_rockwool_financial_mapping.py
```

**Causal-ML and predictive modelling**

```bash
python3 scripts/causal_ml/01_causal_data_audit.py
python3 scripts/causal_ml/02_ml_structure_check.py
python3 scripts/causal_ml/03_build_modeling_panel.py
python3 scripts/causal_ml/04_temporal_model_benchmark.py
python3 scripts/causal_ml/05_temporal_model_benchmark_per_year.py
python3 scripts/causal_ml/06_temporal_model_cpcv_benchmark.py
python3 scripts/causal_ml/07_cpcv_statistical_inference.py
python3 scripts/causal_ml/08_quantile_regression.py
python3 scripts/causal_ml/09_model_explainability.py
python3 scripts/causal_ml/10_causal_dml.py
python3 scripts/causal_ml/11_strict_pre_treatment_dml.py
python3 scripts/causal_ml/12_dml_temporal_robustness.py
python3 scripts/causal_ml/13_dml_heterogeneous_effects.py
```

### 13.3 Execution dependencies

The commands above are grouped by workflow, not a guarantee that every script can run independently.

Important dependencies include:

- Market-data preparation before market indicator construction.
- Macroeconomic preparation before macroeconomic panel construction.
- Construction panel creation before downstream predictive and DML workflows.
- CPCV predictions before CPCV statistical inference.
- Strict DML estimates and predictions before temporal robustness analysis.
- Financial revenue bridge before integrated initiative valuation and reconciliation.
- Financial report downloads and extraction before company financial mapping.

Some scripts may download external data or require source files to be available locally.

### 13.4 Reproducibility principles

- Preserve raw data separately from processed datasets.
- Do not overwrite original source files with transformed data.
- Retain the category mapping used in each modelling workflow.
- Preserve model configurations and random seeds.
- Store generated tables and figures in their designated output directories.
- Record the data period, sample size, validation method, and inference procedure for every reported model result.

---

## 14. Limitations and Interpretation Policy

The project distinguishes among observed facts, statistical associations, predictive results, causal estimates, and scenario-based financial projections.

The following limitations govern interpretation.

### 14.1 Aggregate data

Danish construction statistics do not directly measure:

- Insulation demand.
- ROCKWOOL market share.
- ROCKWOOL project specifications.
- Distributor-level sell-through.
- Commercial conversion rates.

### 14.2 Small annual sample

Several econometric and inferential procedures rely on only 23 annual observations or clusters.

This limits statistical power and makes temporal dependence particularly important.

### 14.3 Temporal validation

CPCV assesses performance across combinations of historical periods but is not equivalent to strict forward-only deployment validation.

### 14.4 Causal identification

The aggregate observational data do not establish conditional exchangeability or the other assumptions required for causal identification.

The DML treatment is a construction-market variable, not a randomized or observed ROCKWOOL intervention.

### 14.5 Financial assumptions

The normalized opportunity pool and conversion rates are scenario assumptions, not observed market-wide commercial data.

### 14.6 Category aggregation

Statistics Denmark building-use categories do not map exactly to ROCKWOOL product applications, customer segments, or commercial territories.

### 14.7 Financial comparability

Company-level financial benchmarks differ in accounting definitions, reporting scopes, and business composition.

### 14.8 Implementation risk

The initiative's projected NPV and IRR depend on achieving the assumed conversion improvements and revenue realization.

### Interpretation policy

The project explicitly separates:

1. **Observed market facts** — directly derived from source datasets.
2. **Statistical associations** — estimated relationships between observed variables.
3. **Predictive performance** — out-of-sample or cross-validated model performance.
4. **Causal estimates** — estimates requiring explicit identification assumptions.
5. **Financial projections** — outputs conditional on scenario assumptions.
6. **Strategic recommendations** — proposed commercial actions informed by the preceding evidence.

These categories must not be presented as interchangeable evidence.

---

## 15. Intended Strategic Deliverable

The final case recommendation integrates the following workstreams:

1. Danish GBI market conditions and construction-cycle evidence.
2. Existing building stock and renovation opportunity.
3. Macroeconomic conditions affecting construction activity.
4. Specification-to-purchase conversion bottlenecks.
5. Commercial initiative design and execution.
6. Revenue conversion scenarios.
7. Initiative financial feasibility and sensitivity.
8. Evidence quality, model uncertainty, and implementation risks.

The objective is to develop a defensible commercial strategy that connects market opportunity to execution, quantifies the potential financial impact, and identifies the data ROCKWOOL would need to validate and operationalize the strategy.

---

## 16. Project Status

| Workstream | Status |
|---|---|
| Danish construction data audit | Completed |
| Macroeconomic data preparation | Completed |
| Market indicator construction | Completed |
| Construction panel reconciliation | Completed |
| Baseline econometric analysis | Completed |
| Econometric robustness and specifications | Completed |
| Company financial report extraction | Completed |
| Financial scenario model | Completed |
| Revenue conversion bridge | Completed |
| Integrated initiative valuation | Completed |
| Financial reconciliation | Completed |
| Funnel sensitivity analysis | Completed |
| Temporal ML benchmarking | Completed |
| CPCV statistical inference | Completed |
| Quantile regression | Completed |
| Expanded-control DML | Completed |
| Strict pre-treatment DML | Completed |
| DML temporal robustness | Completed |
| DML heterogeneous effects | Completed |
| Model explainability | Pending verification |
| Consolidated research report | In progress |
| Final integrated case recommendation | In progress |

---

## 17. Project Metadata

| Field | Description |
|---|---|
| Project | CBS GLOBAL 2026 |
| Case company | ROCKWOOL |
| Market | Danish Glasswool and Building Insulation |
| Geographic scope | Denmark |
| Main analysis period | Varies by dataset |
| Construction activity history | 1982–2025 |
| Macroeconomic overlap | 2002–2025 |
| Main analytical tools | Python, pandas, NumPy, scikit-learn, statsmodels, XGBoost, LightGBM, CatBoost |
| Main research areas | Market intelligence, econometrics, financial modelling, predictive ML, DML |
| Primary objective | Evidence-based commercial growth strategy through 2030 |

---

**End of README**
