
from pathlib import Path
from urllib.request import Request, urlopen
import csv

OUT = Path("data/raw/financials")
OUT.mkdir(parents=True, exist_ok=True)

REPORTS = [
    {
        "company": "ROCKWOOL",
        "year": 2025,
        "file": "rockwool_2025.zip",
        "url": "https://www.rockwool.com/siteassets/investors/financial-reports/2025/rock-2025-12-31-en.zip?f=20260813122012",
        "currency": "EUR"
    },
    {
        "company": "Kingspan",
        "year": 2025,
        "file": "kingspan_2025.pdf",
        "url": "https://annual-report.kingspan.com/wp-content/uploads/2026/03/KS_AR25_Web.pdf",
        "currency": "EUR"
    },
    {
        "company": "Saint-Gobain",
        "year": 2025,
        "file": "saint_gobain_2025.pdf",
        "url": "https://www.saint-gobain.com/sites/saint-gobain.com/files/media/document/Saint-Gobain_2025_DEU_VA.pdf",
        "currency": "EUR"
    },
    {
        "company": "Owens Corning",
        "year": 2025,
        "file": "owens_corning_2025.html",
        "url": "https://www.sec.gov/Archives/edgar/data/1370946/000137094626000067/oc-20251231.htm",
        "currency": "USD"
    },
    {
        "company": "Sto",
        "year": 2025,
        "file": "sto_2025.pdf",
        "url": "https://www.sto.com/media/documents/company_documents/Geschaeftsbericht_Sto_-2025_interaktiv_en.pdf",
        "currency": "EUR"
    }
]

manifest = []

for report in REPORTS:
    path = OUT / report["file"]

    try:
        request = Request(
            report["url"],
            headers={"User-Agent": "Mozilla/5.0 research contact analyst@example.com"}
        )

        with urlopen(request, timeout=120) as response:
            data = response.read()

        path.write_bytes(data)

        manifest.append({
            **report,
            "status": "downloaded",
            "size_bytes": len(data)
        })

        print(f"Downloaded: {report['company']} {report['year']} "
              f"({len(data):,} bytes)")

    except Exception as exc:
        manifest.append({
            **report,
            "status": f"failed: {exc}",
            "size_bytes": 0
        })

        print(f"FAILED: {report['company']} - {exc}")

with open(OUT / "report_manifest.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=manifest[0].keys())
    writer.writeheader()
    writer.writerows(manifest)

print("\nDOWNLOAD SUMMARY")
for item in manifest:
    print(item["company"], item["status"])

print(f"\nReports saved to: {OUT}")
