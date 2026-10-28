"""Recursively shrink CSVs larger than 90 MB by randomly retaining data rows.

Run from anywhere: python limit_csv_to_90mb.py

- Searches the directory containing this script and every subdirectory.
- Keeps the header and the original order of the retained rows.
- Streams CSV records, including records with embedded line breaks.
- Makes an original-file backup OUTSIDE the scanned directory.
- Replaces a source file only after a valid, size-limited output is ready.

NOTE: Random subsampling changes trial counts and other scientific statistics.
Use this only for reduced-size copies, never for the authoritative dataset.
"""

from __future__ import annotations

import codecs
import csv
import os
import random
import shutil
import tempfile
from pathlib import Path

# 90 MB in decimal bytes. The GitHub per-file hard limit is 100 MB.
MAX_SIZE_BYTES = 90_000_000
# Leave a little space below the ceiling to avoid overshooting due to
# differences in row lengths and CSV formatting.
TARGET_FRACTION = 0.985
SEED = 20261028

ROOT = Path(__file__).resolve().parent
# For H:\reports_NAS, this is H:\reports_NAS_original_csv_backup.
# Putting backups outside ROOT ensures they are not included in the scan or Git.
BACKUP_ROOT = ROOT.parent / f"{ROOT.name}_original_csv_backup"


def mb(nbytes: int) -> str:
    return f"{nbytes / 1_000_000:.2f} MB"


def detect_format(path: Path):
    with path.open("rb") as f:
        sample = f.read(65_536)

    encoding = "utf-8-sig" if sample.startswith(codecs.BOM_UTF8) else "utf-8"
    sample_text = sample.decode(encoding, errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel  # Standard comma-separated CSV fallback.

    if b"\r\n" in sample:
        line_ending = "\r\n"
    elif b"\n" in sample:
        line_ending = "\n"
    elif b"\r" in sample:
        line_ending = "\r"
    else:
        line_ending = "\n"

    return encoding, dialect, line_ending


def create_temp_file(parent: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=".csv_sample_", suffix=".tmp", dir=parent)
    os.close(fd)
    return Path(name)


def sample_csv(source: Path, destination: Path, probability: float,
               encoding: str, dialect, line_ending: str,
               rng: random.Random) -> tuple[int, int]:
    kept = dropped = 0
    with (
        source.open("r", encoding=encoding, newline="") as fin,
        destination.open("w", encoding=encoding, newline="") as fout,
    ):
        reader = csv.reader(fin, dialect=dialect, strict=True)
        writer = csv.writer(fout, dialect=dialect, lineterminator=line_ending)

        header = next(reader, None)
        if header is None:
            return kept, dropped
        writer.writerow(header)  # Never discard column names.

        for record in reader:
            if rng.random() < probability:
                writer.writerow(record)
                kept += 1
            else:
                dropped += 1

    return kept, dropped


def shrink_file(path: Path) -> bool:
    original_size = path.stat().st_size
    if original_size <= MAX_SIZE_BYTES:
        return False

    relative_path = path.relative_to(ROOT)
    backup_path = BACKUP_ROOT / relative_path
    print(f"\nProcessing: {relative_path} ({mb(original_size)})")

    encoding, dialect, line_ending = detect_format(path)
    rng = random.Random(f"{SEED}:{relative_path.as_posix()}")
    temp_paths: list[Path] = []
    current = path
    total_dropped = 0
    try:
        for attempt in range(1, 11):
            current_size = current.stat().st_size
            probability = min(1.0, MAX_SIZE_BYTES * TARGET_FRACTION / current_size)
            output = create_temp_file(path.parent)
            temp_paths.append(output)
            kept, dropped = sample_csv(
                current, output, probability, encoding, dialect, line_ending, rng
            )
            total_dropped += dropped
            new_size = output.stat().st_size
            print(f"  Pass {attempt}: retained {kept:,} rows; output {mb(new_size)}")

            if new_size <= MAX_SIZE_BYTES:
                # Back up the full current original before replacing it.
                # If an earlier backup exists, use a new numbered filename.
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                available_backup = backup_path
                counter = 1
                while available_backup.exists():
                    available_backup = backup_path.with_name(
                        f"{backup_path.stem}_backup_{counter}{backup_path.suffix}"
                    )
                    counter += 1
                shutil.copy2(path, available_backup)
                if available_backup.stat().st_size != original_size:
                    raise OSError("Backup size does not match original")

                os.replace(output, path)  # Atomic replacement on same disk.
                print(f"  Done: {mb(original_size)} -> {mb(new_size)}")
                print(f"  Backup: {available_backup}")
                return True

            current = output  # Try another random reduction if still too big.

        raise RuntimeError("Could not get below 90 MB in 10 passes; original unchanged")
    finally:
        for temp in temp_paths:
            temp.unlink(missing_ok=True)


def main() -> None:
    print(f"Scanning recursively: {ROOT}")
    print(f"Maximum CSV size: {mb(MAX_SIZE_BYTES)}")
    print(f"Backups outside repository: {BACKUP_ROOT}")

    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
        and ".git" not in path.relative_to(ROOT).parts
    )
    resized = failed = 0
    for path in files:
        try:
            resized += shrink_file(path)
        except (OSError, UnicodeError, csv.Error, RuntimeError) as exc:
            failed += 1
            print(f"  ERROR: {path.relative_to(ROOT)}: {exc}")
            print("  Original file was not replaced.")

    print(f"\nFinished. Found {len(files)} CSV files; resized {resized}; failures {failed}.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    # Increase limit for unusually long CSV fields, without loading full files.
    csv.field_size_limit(100_000_000)
    main()
