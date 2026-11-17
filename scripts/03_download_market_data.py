
import io
import json
from pathlib import Path

import pandas as pd
import requests

API = "https://api.statbank.dk/v1/data"

TABLES = {
    "BYGV01": {},
    "BYGV11": {
        "OMRÅDE": ["000"]
    },
    "BYGB34": {
        "OMRÅDE": ["000"]
    },
    "BYGB40": {
        "OMRÅDE": ["000"],
        "MÆNGDE4": ["50"]
    },
    "BYG43": {},
    "PRIS91": {}
}

METADATA = Path("data/raw/metadata")
RAW = Path("data/raw/statbank")
PROCESSED = Path("data/processed")

RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

session = requests.Session()

for table, overrides in TABLES.items():
    print(f"\nDownloading {table}...")

    metadata_path = METADATA / f"{table.lower()}.json"

    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)

    variables = []

    for variable in metadata["variables"]:
        code = variable["id"]

        selected_values = overrides.get(code, ["*"])

        variables.append({
            "code": code,
            "values": selected_values
        })

    payload = {
        "table": table,
        "format": "CSV",
        "lang": "en",
        "timeOrder": "Ascending",
        "valuePresentation": "Value",
        "variables": variables
    }

    response = session.post(
        API,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    if not response.text.strip():
        raise ValueError(f"{table}: Empty API response")

    raw_path = RAW / f"{table.lower()}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(response.text, encoding="utf-8-sig")

    df = pd.read_csv(
        io.StringIO(response.text),
        sep=";",
        decimal=","
    )

    if df.empty:
        raise ValueError(f"{table}: No observations returned")

    processed_path = PROCESSED / f"{table.lower()}.csv"

    df.to_csv(
        processed_path,
        index=False
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Raw: {raw_path}")
    print(f"Processed: {processed_path}")
    print(df.head(3).to_string(index=False))

print("\nMarket data acquisition completed.")
