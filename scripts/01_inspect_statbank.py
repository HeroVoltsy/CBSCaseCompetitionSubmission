
import json
from pathlib import Path

import requests

TABLES = [
    "BYGV01",
    "BYGV11",
    "BYGB34",
    "BYGB40",
    "BYG43",
    "PRIS91"
]

BASE_URL = "https://api.statbank.dk/v1/tableinfo"

Path("data/raw/metadata").mkdir(parents=True, exist_ok=True)

for table in TABLES:
    response = requests.post(
        BASE_URL,
        json={"table": table, "lang": "en"}
    )
    response.raise_for_status()

    metadata = response.json()

    output = Path("data/raw/metadata") / f"{table.lower()}.json"

    with output.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"{table}: metadata saved")
