
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import pandas as pd
import re

ZIP = Path("data/raw/financials/rockwool_2025.zip")
OUT = Path("data/processed/financials")
OUT.mkdir(parents=True, exist_ok=True)

def local_name(element):
    return etree.QName(element).localname

def clean_number(value, scale, sign):
    value = re.sub(r"\s+", "", value or "")
    value = value.replace(",", "").replace("\u00a0", "")

    if not value or value in {"-", "—", "–"}:
        return None

    negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()")

    try:
        number = float(value)
        number *= 10 ** int(scale or 0)

        if negative or sign == "-":
            number = -abs(number)

        return number
    except ValueError:
        return None

with ZipFile(ZIP) as archive:
    report_name = next(
        name for name in archive.namelist()
        if name.endswith(".xhtml")
    )

    root = etree.fromstring(archive.read(report_name))

contexts = {}
units = {}

for element in root.iter():
    tag = local_name(element)

    if tag == "context":
        context_id = element.get("id")
        period = element.find(".//{*}period")

        start = period.findtext("{*}startDate") if period is not None else None
        end = period.findtext("{*}endDate") if period is not None else None
        instant = period.findtext("{*}instant") if period is not None else None

        dimensions = []

        for child in element.iter():
            if local_name(child) in {"explicitMember", "typedMember"}:
                dimensions.append(
                    f"{child.get('dimension')}={''.join(child.itertext()).strip()}"
                )

        contexts[context_id] = {
            "period_start": start,
            "period_end": end,
            "period_instant": instant,
            "dimensions": "|".join(dimensions)
        }

    elif tag == "unit":
        unit_id = element.get("id")
        units[unit_id] = "|".join(
            "".join(child.itertext()).strip()
            for child in element
        )

facts = []

for element in root.iter():
    tag = local_name(element)

    if tag not in {"nonFraction", "nonNumeric", "fraction"}:
        continue

    context_id = element.get("contextRef")
    context = contexts.get(context_id, {})

    raw_value = "".join(element.itertext()).strip()
    scale = element.get("scale")
    sign = element.get("sign")

    facts.append({
        "concept": element.get("name"),
        "context_ref": context_id,
        "unit_ref": element.get("unitRef"),
        "unit": units.get(element.get("unitRef")),
        "fact_type": tag,
        "raw_value": raw_value,
        "numeric_value": (
            clean_number(raw_value, scale, sign)
            if tag == "nonFraction" else None
        ),
        "scale": scale,
        "sign": sign,
        "decimals": element.get("decimals"),
        "period_start": context.get("period_start"),
        "period_end": context.get("period_end"),
        "period_instant": context.get("period_instant"),
        "dimensions": context.get("dimensions")
    })

facts = pd.DataFrame(facts)

facts.to_csv(
    OUT / "rockwool_xbrl_facts.csv",
    index=False
)

pd.DataFrame([
    {"context_ref": key, **value}
    for key, value in contexts.items()
]).to_csv(
    OUT / "rockwool_xbrl_contexts.csv",
    index=False
)

print("XBRL EXTRACTION COMPLETE")
print(f"Facts: {len(facts):,}")
print(f"Contexts: {len(contexts):,}")
print(f"Units: {len(units):,}")
print(facts["concept"].nunique(), "unique concepts")
print(f"\nSaved to: {OUT}")
print(
    facts.groupby("fact_type").size().to_string()
)
