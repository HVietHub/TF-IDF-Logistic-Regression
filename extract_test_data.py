"""
JSONL to CSV Data Extraction Script
====================================
Extracts and converts datasets from JSONL format (e.g. test.jsonl) into CSV format (test.csv)
for the TF-IDF prompt injection & jailbreak detection pipeline.

Mapping Logic:
  - benign (attack_family is None) -> 'benign' (label_id: 0)
  - attack + attack_family 'PI'    -> 'promptinjection' (label_id: 1)
  - attack + attack_family 'JB'    -> 'jailbreak' (label_id: 2)
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_ID_MAP = {
    "benign": 0,
    "promptinjection": 1,
    "jailbreak": 2,
}


def map_record(record: Dict) -> Dict:
    """Map a single JSONL record to standardized classification fields."""
    raw_label = str(record.get("label", "")).strip().lower()
    raw_family = record.get("attack_family")
    family_str = str(raw_family).strip().upper() if raw_family is not None else ""

    # Determine 3-class label
    if raw_label == "benign" or raw_label == "0":
        standard_label = "benign"
        binary_label = "benign"
    elif raw_label in ("attack", "1", "malicious"):
        binary_label = "attack"
        if family_str == "PI":
            standard_label = "promptinjection"
        elif family_str == "JB":
            standard_label = "jailbreak"
        else:
            # Fallback if attack family unspecified
            standard_label = "jailbreak"
    else:
        # Fallback by family
        if family_str == "PI":
            standard_label = "promptinjection"
            binary_label = "attack"
        elif family_str == "JB":
            standard_label = "jailbreak"
            binary_label = "attack"
        else:
            standard_label = "benign"
            binary_label = "benign"

    label_id = LABEL_ID_MAP[standard_label]

    return {
        "id": record.get("id", ""),
        "text": record.get("text", ""),
        "label": standard_label,
        "label_id": label_id,
        "attack_family": raw_family if raw_family is not None else "None",
        "binary_label": binary_label,
    }


def extract_jsonl_to_csv(
    input_path: str,
    output_path: str,
    format_mode: str = "standard",
    backup_existing: bool = True,
) -> pd.DataFrame:
    """Extract records from JSONL and export to CSV.
    
    Args:
        input_path: Path to input .jsonl file.
        output_path: Path to output .csv file.
        format_mode: 'standard' (text, label), 'full' (all metadata columns), 'binary' (text, label with benign/attack).
        backup_existing: Create a backup of existing destination file before overwriting.
        
    Returns:
        Processed pandas DataFrame.
    """
    in_file = Path(input_path)
    out_file = Path(output_path)

    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Reading JSONL from: {in_file}")
    records = []
    skipped = 0

    with open(in_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                processed = map_record(data)
                records.append(processed)
            except Exception as e:
                logger.warning(f"Line {line_num}: JSON parse error: {e}")
                skipped += 1

    df = pd.DataFrame(records)
    total_records = len(df)
    logger.info(f"Loaded {total_records} valid records (skipped: {skipped}).")

    # Select columns based on format mode
    if format_mode == "standard":
        # Exactly matches the existing test.csv schema (text, label)
        export_df = df[["text", "label"]].copy()
    elif format_mode == "binary":
        export_df = df[["text", "binary_label"]].rename(columns={"binary_label": "label"}).copy()
    elif format_mode == "full":
        export_df = df[["id", "text", "label", "label_id", "attack_family", "binary_label"]].copy()
    else:
        raise ValueError(f"Unknown format mode: {format_mode}. Choose 'standard', 'full', or 'binary'.")

    # Backup existing file if requested
    if backup_existing and out_file.exists():
        backup_path = out_file.parent / f"{out_file.stem}_backup{out_file.suffix}"
        shutil.copy2(out_file, backup_path)
        logger.info(f"Backed up existing file to: {backup_path}")

    # Export to CSV
    out_file.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out_file, index=False, encoding="utf-8")
    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(export_df)} rows to: {out_file} ({file_size_mb:.2f} MB)")

    # Print summary
    print("\n" + "=" * 65)
    print("      DATA EXTRACTION SUMMARY (JSONL -> CSV)")
    print("=" * 65)
    print(f"Input File:       {in_file}")
    print(f"Output File:      {out_file}")
    print(f"Total Rows:       {len(export_df):,}")
    print(f"Columns Exported: {export_df.columns.tolist()}")
    print("-" * 65)
    print("Class Distribution (3-Class Standard):")
    counts_3class = df["label"].value_counts()
    for lbl, count in counts_3class.items():
        pct = (count / total_records) * 100
        print(f"  - {lbl:<16}: {count:>5} ({pct:>5.1f}%)")
    print("-" * 65)
    print("Attack Family Breakdown:")
    counts_family = df["attack_family"].value_counts()
    for fam, count in counts_family.items():
        pct = (count / total_records) * 100
        fam_name = "Benign (None)" if fam == "None" else f"Attack ({fam})"
        print(f"  - {fam_name:<16}: {count:>5} ({pct:>5.1f}%)")
    print("-" * 65)
    print("Sample Rows (First 3):")
    for idx, row in export_df.head(3).iterrows():
        sample_text = str(row["text"])[:80].replace("\n", " ")
        print(f"  [{idx + 1}] Label: {row['label']:<16} | Text: {sample_text}...")
    print("=" * 65 + "\n")

    return export_df


def main():
    parser = argparse.ArgumentParser(description="Extract and convert test.jsonl to test.csv")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/test.jsonl",
        help="Path to input .jsonl file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/test.csv",
        help="Path to output .csv file",
    )
    parser.add_argument(
        "--format",
        "-f",
        type=str,
        default="standard",
        choices=["standard", "full", "binary"],
        help="Output CSV format: 'standard' (text, label), 'full' (all metadata), 'binary' (text, benign/attack)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup existing output file",
    )

    args = parser.parse_args()

    extract_jsonl_to_csv(
        input_path=args.input,
        output_path=args.output,
        format_mode=args.format,
        backup_existing=not args.no_backup,
    )


if __name__ == "__main__":
    main()
