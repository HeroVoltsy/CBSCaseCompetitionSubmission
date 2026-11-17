
import json
from pathlib import Path

TABLES = [
    "bygv01",
    "bygv11",
    "bygb34",
    "bygb40",
    "byg43",
    "pris91"
]

BASE = Path("data/raw/metadata")

for table in TABLES:
    path = BASE / f"{table}.json"

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    print("\n" + "=" * 75)
    print(f"TABLE: {table.upper()}")
    print("=" * 75)

    print("Description:", data.get("description", "N/A"))
    print("Unit:", data.get("unit", "N/A"))

    variables = data.get("variables", [])

    for variable in variables:
        print(f"\nDimension: {variable.get('text')}")
        print(f"ID: {variable.get('id')}")

        values = variable.get("values", [])
        print(f"Available values: {len(values)}")

        for value in values[:12]:
            print(
                f"  {value.get('id')} | "
                f"{value.get('text')}"
            )

        if len(values) > 12:
            print(f"  ... {len(values) - 12} more")
