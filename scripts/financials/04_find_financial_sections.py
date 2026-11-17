
from pathlib import Path
import re

BASE = Path("data/processed/financials/extracted")

PATTERNS = [
    r"consolidated income statement",
    r"consolidated statement of profit or loss",
    r"consolidated statement of financial position",
    r"consolidated balance sheet",
    r"consolidated cash flow statement",
    r"consolidated statement of cash flows",
    r"revenue",
    r"operating profit",
    r"operating income",
    r"EBITDA",
    r"capital expenditure",
    r"cash generated from operations",
    r"net debt",
    r"total assets"
]

for path in sorted(BASE.glob("*.txt")):
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    print(f"\n{'=' * 75}")
    print(path.name)

    matches = []

    for i, line in enumerate(lines):
        if any(re.search(p, line, re.I) for p in PATTERNS):
            matches.append(i)

    selected = set()

    for i in matches:
        for j in range(max(0, i - 2), min(len(lines), i + 4)):
            selected.add(j)

    output = BASE / f"{path.stem}_financial_excerpts.txt"

    with output.open("w") as file:
        previous = -2

        for i in sorted(selected):
            if i > previous + 1:
                file.write("\n...\n")

            file.write(f"{i + 1}: {lines[i]}\n")
            previous = i

    print(f"Lines: {len(lines):,}")
    print(f"Matches: {len(matches):,}")
    print(f"Excerpts: {output}")
