
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import shutil
import subprocess

BASE = Path("data/raw/financials")
OUT = Path("data/processed/financials")
TEXT = OUT / "extracted"

TEXT.mkdir(parents=True, exist_ok=True)

for path in sorted(BASE.iterdir()):
    if path.suffix.lower() not in {".pdf", ".zip", ".html"}:
        continue

    print(f"\n{'=' * 65}")
    print(path.name)
    print(f"Size: {path.stat().st_size:,} bytes")

    with path.open("rb") as file:
        signature = file.read(8)

    print(f"Signature: {signature!r}")

    if path.suffix.lower() == ".pdf":
        if shutil.which("pdfinfo"):
            result = subprocess.run(
                ["pdfinfo", str(path)],
                capture_output=True,
                text=True
            )
            print(result.stdout[:2500] or result.stderr[:1000])

        if shutil.which("pdftotext"):
            output = TEXT / f"{path.stem}.txt"
            subprocess.run(
                ["pdftotext", "-layout", str(path), str(output)],
                check=True
            )
            print(f"Extracted text: {output}")

    elif path.suffix.lower() == ".zip":
        try:
            with ZipFile(path) as archive:
                files = archive.infolist()
                print(f"Archive files: {len(files)}")

                for item in files[:30]:
                    print(item.filename, f"({item.file_size:,} bytes)")

                archive.extractall(TEXT / path.stem)
                print(f"Extracted archive: {TEXT / path.stem}")

        except BadZipFile:
            print("INVALID ZIP ARCHIVE")

    elif path.suffix.lower() == ".html":
        output = TEXT / f"{path.stem}.txt"

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(
                path.read_text(errors="ignore"),
                "html.parser"
            )

            output.write_text(
                soup.get_text(separator="\n", strip=True)
            )

            print(f"Extracted text: {output}")

        except ImportError:
            print("Install beautifulsoup4 to extract HTML text")

print("\nINSPECTION COMPLETE")

