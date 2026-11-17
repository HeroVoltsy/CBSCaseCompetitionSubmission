
#!/usr/bin/env python3

from pathlib import Path
import argparse
import os

ROOT = Path(__file__).resolve().parent.parent

MARKER = "<!-- AUTO-GENERATED README: DO NOT EDIT BETWEEN GENERATIONS -->"

# ---------------------------------------------------------------------
# DIRECTORY DOCUMENTATION MANIFEST
# ---------------------------------------------------------------------

DOCS = {
    ".": {
        "title": "CBS GLOBAL 2026 — ROCKWOOL Denmark",
        "description": "Research repository covering Danish GBI market intelligence, econometrics, financial modelling, predictive machine learning, and causal analysis.",
        "sections": [
            ("Research objective", """
Evaluate how ROCKWOOL can achieve 20% revenue growth in the Danish
Glasswool and Building Insulation market by 2030.

The analysis combines Danish construction statistics, macroeconomic
indicators, company financial disclosures, scenario-based financial
modelling, predictive benchmarking, and double machine learning.
"""),
            ("Commercial hypothesis", """
> ROCKWOOL's Danish GBI growth constraint is not simply insufficient
> market demand; it is the failure to consistently convert technical
> preference into commercially secured purchases across the Danish
> building-material value chain.

The proposed strategy is to win the Danish renovation journey by
securing conversion from specification to purchase.

The hypothesized mechanism is:

1. Early technical engagement.
2. Specification capture.
3. Specification retention.
4. Distributor coordination.
5. Purchase conversion.
6. Incremental revenue.
"""),
            ("Statistical causal hypothesis", """
The DML workstream investigates a separate market-level hypothesis:

> Conditional on prior construction activity, building-use category,
> and deterministic time trends, an increase in building permits in
> the preceding year increases subsequent construction starts.

Treatment: `Permits_lag1`

Outcome: `Started`

Pre-treatment control: `Started_lag2`

The statistical hypothesis is not equivalent to the commercial
hypothesis. The available aggregate data do not identify the causal
effect of ROCKWOOL-specific commercial interventions.
"""),
            ("Causal identification assumptions", """
A causal interpretation of the DML estimates requires:

1. **Temporal ordering and consistency:** Lagged permits precede
   current construction starts, and treatment/outcome definitions
   remain consistent.
2. **Conditional exchangeability:** No unobserved confounding remains
   after conditioning on the specified pre-treatment controls.
3. **Overlap:** Sufficient residual treatment variation exists
   conditional on observed covariates.
4. **No relevant interference:** Cross-category treatment spillovers
   do not invalidate the estimand.
5. **Nuisance estimation:** Conditional treatment and outcome
   functions are estimated sufficiently well for orthogonalization.
6. **Valid inference:** Dependence across observations and years is
   appropriately reflected in uncertainty estimates.

These assumptions cannot all be verified using the available
aggregate observational data. Therefore, the DML findings should be
interpreted cautiously as conditional relationships rather than
definitive causal proof.
"""),
            ("Case parameters", """
| Parameter | Assumption |
|---|---:|
| Baseline annual revenue | €60m |
| 2030 revenue target | €72m |
| Required growth | 20% |
| Incremental contribution margin | 25% |
| Annual initiative budget | €135,000 |
| Investment period | 2027–2030 |
| Total initiative budget | €540,000 |

These are illustrative case parameters, not actual Danish ROCKWOOL
segment disclosures.
"""),
            ("Research findings", """
- Danish construction starts peaked at approximately 9.98 million m²
  in 2021 and declined to approximately 5.08 million m² in 2025.
- The econometric baseline identifies macroeconomic associations
  with aggregate construction-start growth, subject to a small
  annual sample and influential observations.
- Persistence is difficult to outperform under the implemented
  CPCV benchmark. No trained model demonstrates a statistically
  significant overall improvement after multiple-testing correction.
- Quantile regression identifies positive conditional associations
  between lagged permits and starts.
- Strict pre-treatment DML produces positive estimates under Ridge
  and ExtraTrees nuisance models.
- DML heterogeneity tests do not reject equality of coefficients
  across the four broad building-use groups.
- The base financial scenario produces approximately €72.069m in
  2030 revenue and an NPV of approximately €5.128m at a 10% discount
  rate. These are assumption-dependent projections.
"""),
            ("Methodological limitations", """
- Construction statistics do not directly measure insulation demand
  or ROCKWOOL market share.
- The DML analysis does not estimate the effect of ROCKWOOL commercial
  interventions.
- The annual inference sample is small.
- CPCV is not equivalent to strict forward-only validation.
- The financial opportunity pool and conversion rates are model
  assumptions rather than independently observed market quantities.
- Global company financial benchmarks are not directly comparable
  across accounting definitions and business mixes.
"""),
            ("Execution", """
```bash
# Market data and preparation
python3 scripts/01_inspect_statbank.py
python3 scripts/02_summarize_metadata.py
python3 scripts/03_download_market_data.py
python3 scripts/04_market_analysis.py
python3 scripts/05_prepare_macro.py
python3 scripts/06_construction_indicators.py

# Econometrics and financial modelling
python3 scripts/07_econometric_baseline.py
python3 scripts/08_econometric_effects.py
python3 scripts/09_initiative_financial_model.py
python3 scripts/10_econometric_robustness.py
python3 scripts/11_econometric_specifications.py
python3 scripts/12_revenue_conversion_bridge.py
python3 scripts/13_integrated_initiative_valuation.py
python3 scripts/14_financial_reconciliation.py
python3 scripts/15_funnel_sensitivity.py

# Causal ML
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
"""),
        ],
    },

    "data": {
        "title": "Data",
        "description": "All source datasets, external evidence, and processed analytical data.",
    },
    "data/raw": {
        "title": "Raw Data",
        "description": "Original downloaded source datasets and documents. Preserve these files as immutable inputs.",
    },
    "data/raw/statbank": {
        "title": "Statistics Denmark — Raw Data",
        "description": "Original StatBank CSV exports used for Danish construction and building-stock analysis.",
    },
    "data/raw/metadata": {
        "title": "StatBank Metadata",
        "description": "Source metadata describing dimensions, categories, units, and table structures.",
    },
    "data/raw/macroeconomic": {
        "title": "Raw Macroeconomic Data",
        "description": "Original Danish GDP, CPI, inflation, and lending-rate datasets.",
    },
    "data/raw/financials": {
        "title": "Raw Financial Disclosures",
        "description": "Original annual reports, XBRL packages, and downloaded company financial documents.",
    },
    "data/external_evidence": {
        "title": "External Evidence",
        "description": "External research, industry evidence, policy material, and supporting references used in the case analysis.",
    },
    "data/processed": {
        "title": "Processed Data",
        "description": "Cleaned datasets, derived indicators, model inputs, and generated analytical results.",
    },
    "data/processed/market_indicators": {
        "title": "Market Indicators",
        "description": "Annual and quarterly Danish construction activity, building stock, construction costs, and renovation price indicators.",
    },
    "data/processed/macroeconomic": {
        "title": "Processed Macroeconomic Data",
        "description": "Prepared annual Danish macroeconomic indicators and aligned time-series data.",
    },
    "data/processed/modeling": {
        "title": "Econometric Outputs",
        "description": "Baseline regressions, effect sizes, model diagnostics, robustness checks, and alternative specifications.",
    },
    "data/processed/financial_model": {
        "title": "Financial Model Outputs",
        "description": "Revenue conversion bridge, scenario modelling, initiative valuation, break-even analysis, and sensitivity results.",
    },
    "data/processed/financial_model/funnel_sensitivity": {
        "title": "Funnel Sensitivity",
        "description": "One-way and factorial sensitivity analysis of specification capture, specification retention, and purchase conversion.",
    },
    "data/processed/financial_model/reconciliation": {
        "title": "Financial Reconciliation",
        "description": "Revenue trajectory, cash-flow, NPV, and sensitivity reconciliation outputs.",
    },
    "data/processed/financials": {
        "title": "Processed Company Financials",
        "description": "Company benchmarks, ROCKWOOL XBRL facts, financial contexts, and extracted financial mappings.",
    },
    "data/processed/financials/extracted": {
        "title": "Extracted Financial Disclosures",
        "description": "Text extracts and financial excerpts from company annual reports.",
    },
    "data/processed/causal_ml": {
        "title": "Causal ML and Predictive Modelling",
        "description": "Construction modelling panels, predictive benchmarks, statistical inference, quantile regression, explainability, and DML results.",
    },
    "data/processed/causal_ml/cpcv": {
        "title": "CPCV Results",
        "description": "Combinatorial temporal validation predictions, split metrics, model comparisons, and paired statistical inference.",
    },
    "data/processed/causal_ml/cpcv/inference": {
        "title": "CPCV Statistical Inference",
        "description": "Annual metrics, paired model comparisons, category-level effects, prediction correlations, and residual correlations.",
    },
    "data/processed/causal_ml/quantile_regression": {
        "title": "Quantile Regression",
        "description": "Quantile coefficients, model fit statistics, bootstrap estimates, and coefficient-difference results.",
    },
    "data/processed/causal_ml/explainability": {
        "title": "Model Explainability",
        "description": "Grouped temporal permutation importance and SHAP outputs. Interpret only after validation and output verification.",
    },
    "data/processed/causal_ml/dml": {
        "title": "Double Machine Learning",
        "description": "Expanded-control DML estimates, out-of-fold predictions, strict pre-treatment estimates, and heterogeneous-effect analysis.",
    },
    "data/processed/causal_ml/dml/strict_pre_treatment": {
        "title": "Strict Pre-treatment DML",
        "description": "DML estimates and cross-fitted predictions using lagged construction activity and pre-treatment temporal controls.",
    },
    "data/processed/causal_ml/dml/strict_pre_treatment/inference": {
        "title": "DML Temporal Inference",
        "description": "Leave-one-year-out sensitivity, year-level contributions, and moving-block bootstrap results.",
    },
    "data/processed/causal_ml/dml/heterogeneous_effects": {
        "title": "DML Heterogeneous Effects",
        "description": "Building-use category mapping, group-specific estimates, omnibus tests, pairwise comparisons, and block-bootstrap results.",
    },
    "scripts": {
        "title": "Analysis Scripts",
        "description": "Reproducible Python workflows for market analysis, econometrics, financial modelling, and data preparation.",
    },
    "scripts/financials": {
        "title": "Financial Analysis Scripts",
        "description": "Annual-report downloading, inspection, XBRL extraction, financial-section identification, and company benchmark mapping.",
    },
    "scripts/causal_ml": {
        "title": "Causal ML Scripts",
        "description": "Data audits, modelling-panel construction, predictive benchmarking, CPCV, quantile regression, explainability, and DML.",
    },
    "outputs": {
        "title": "Generated Outputs",
        "description": "Presentation-ready figures, tables, and analytical exports.",
    },
    "outputs/figures": {
        "title": "Figures",
        "description": "Visualizations of construction activity, building stock, construction costs, and renovation price indices.",
    },
    "outputs/tables": {
        "title": "Tables",
        "description": "Tabular market-analysis outputs and diagnostic summaries.",
    },
    "outputs/econometrics": {
        "title": "Econometric Outputs",
        "description": "Generated econometric presentation artifacts and analytical exports.",
    },
    "reports": {
        "title": "Research Reports",
        "description": "Consolidated analytical reports and research documentation.",
    },
    "reports/causal_ml": {
        "title": "Causal ML Reports",
        "description": "Consolidated documentation of predictive modelling, statistical inference, and causal analysis.",
    },
}


# ---------------------------------------------------------------------
# FILE INVENTORY
# ---------------------------------------------------------------------

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}

def inventory(directory):
    files = []

    if not directory.exists():
        return files

    for path in sorted(directory.iterdir()):
        if path.name == "README.md":
            continue

        if path.name in IGNORE_DIRS:
            continue

        if path.is_dir():
            count = sum(
                1 for p in path.rglob("*")
                if p.is_file()
                and p.name != "README.md"
                and not any(part in IGNORE_DIRS for part in p.parts)
            )
            files.append(
                f"- `{path.name}/` — directory ({count} files)"
            )
        elif path.is_file():
            size = path.stat().st_size
            files.append(
                f"- `{path.name}` — {format_size(size)}"
            )

    return files


def format_size(size):
    if size < 1024:
        return f"{size} B"
    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 ** 2):.2f} MB"


# ---------------------------------------------------------------------
# README GENERATION
# ---------------------------------------------------------------------

def generate_readme(relative_path, metadata):
    directory = ROOT / relative_path
    title = metadata["title"]
    description = metadata["description"]

    lines = [
        f"# {title}",
        "",
        MARKER,
        "",
        description,
        "",
    ]

    if relative_path != ".":
        lines.extend([
            "## Contents",
            "",
        ])

        contents = inventory(directory)

        if contents:
            lines.extend(contents)
        else:
            lines.append("*No files currently present.*")

        lines.append("")

        lines.extend([
            "## Directory purpose",
            "",
            description,
            "",
        ])

        parent = Path(relative_path).parent

        if str(parent) != ".":
            parent_readme = (
                Path("..") if len(Path(relative_path).parts) == 1
                else Path(*([".."] * len(Path(relative_path).parts)))
            )

            lines.extend([
                "## Related documentation",
                "",
                f"- Parent directory: `{parent.as_posix()}/`",
                "",
            ])

    for heading, content in metadata.get("sections", []):
        lines.extend([
            f"## {heading}",
            "",
            content.strip(),
            "",
        ])

    lines.extend([
        "---",
        "",
        "<!-- This file is generated by scripts/generate_readmes.py -->",
        "",
    ])

    return "\n".join(lines)


def should_write(path, force):
    if not path.exists():
        return True

    existing = path.read_text(encoding="utf-8")

    if MARKER in existing:
        return True

    return force


def main():
    parser = argparse.ArgumentParser(
        description="Generate repository-wide README documentation."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite manually edited README files."
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which README files are missing or outdated."
    )

    args = parser.parse_args()

    generated = 0
    skipped = 0

    for relative_path, metadata in DOCS.items():
        directory = ROOT if relative_path == "." else ROOT / relative_path

        if not directory.exists():
            continue

        output = directory / "README.md"
        content = generate_readme(relative_path, metadata)

        if args.check:
            if not output.exists():
                print(f"MISSING: {output.relative_to(ROOT)}")
            else:
                print(f"EXISTS:  {output.relative_to(ROOT)}")
            continue

        if not should_write(output, args.force):
            print(f"SKIPPED (manual file): {output.relative_to(ROOT)}")
            skipped += 1
            continue

        output.write_text(content, encoding="utf-8")

        print(f"GENERATED: {output.relative_to(ROOT)}")
        generated += 1

    print("\nDOCUMENTATION SUMMARY")
    print(f"Generated: {generated}")
    print(f"Skipped:   {skipped}")


if __name__ == "__main__":
    main()
