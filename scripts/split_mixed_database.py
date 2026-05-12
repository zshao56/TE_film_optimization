import argparse
import csv
import os
import shutil
import sys
from collections import Counter


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_io.metadata import REQUIRED_COLUMNS


def _is_record_start(value):
    value = str(value or "")
    return value.startswith(("sim_", "inverse_"))


def _row_to_records(row, header, line_number):
    if len(row) == len(REQUIRED_COLUMNS):
        chunks = [row]
        fieldnames = REQUIRED_COLUMNS
    elif len(row) == len(header):
        chunks = [row]
        fieldnames = header
    elif len(row) < len(header):
        chunks = [row + [""] * (len(header) - len(row))]
        fieldnames = header
    elif len(row) < len(REQUIRED_COLUMNS):
        chunks = [row + [""] * (len(REQUIRED_COLUMNS) - len(row))]
        fieldnames = REQUIRED_COLUMNS
    else:
        starts = [idx for idx, value in enumerate(row) if _is_record_start(value)]
        if starts and starts[0] != 0:
            starts.insert(0, 0)
        if len(starts) <= 1:
            raise ValueError(
                f"Line {line_number} has {len(row)} fields, which is longer than "
                f"the expanded schema ({len(REQUIRED_COLUMNS)}), and no embedded "
                "simulation_id boundary was found."
            )

        starts.append(len(row))
        chunks = []
        for start, end in zip(starts, starts[1:]):
            chunk = row[start:end]
            if not chunk or all(value == "" for value in chunk):
                continue
            if len(chunk) > len(REQUIRED_COLUMNS):
                print(
                    f"Warning: line {line_number} embedded chunk starting at field {start + 1} "
                    f"has {len(chunk)} fields; truncating to {len(REQUIRED_COLUMNS)}."
                )
                chunk = chunk[:len(REQUIRED_COLUMNS)]
            elif len(chunk) < len(REQUIRED_COLUMNS):
                chunk = chunk + [""] * (len(REQUIRED_COLUMNS) - len(chunk))
            chunks.append(chunk)
        fieldnames = REQUIRED_COLUMNS
        print(f"Warning: repaired line {line_number} by splitting it into {len(chunks)} records.")

    for chunk in chunks:
        record = dict(zip(fieldnames, chunk))
        record["_source_line_number"] = line_number
        yield record


def _read_mixed_rows(metadata_path):
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line_number, row in enumerate(reader, start=2):
            if not row or all(value == "" for value in row):
                continue

            yield from _row_to_records(row, header, line_number)


def _profile_for(record):
    profile = str(record.get("database_profile") or "").strip()
    if profile:
        return profile
    return "legacy"


def _source_field_path(record, source_fields_dir):
    field_file = str(record.get("field_file") or "").strip()
    if not field_file:
        return None
    if os.path.exists(field_file):
        return field_file
    return os.path.join(source_fields_dir, os.path.basename(field_file))


def _copy_or_link(src, dst, use_links):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if use_links:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def split_database(args):
    metadata_path = os.path.abspath(args.metadata)
    source_fields_dir = os.path.abspath(args.source_fields)
    output_root = os.path.abspath(args.output_root)

    records = list(_read_mixed_rows(metadata_path))
    profile_counts = Counter(_profile_for(record) for record in records)
    print("Detected rows by profile:")
    for profile, count in sorted(profile_counts.items()):
        print(f"- {profile}: {count}")

    if args.dry_run:
        return

    selected_profiles = sorted(profile_counts) if args.profile == "all" else [args.profile]
    for profile in selected_profiles:
        selected = [record for record in records if _profile_for(record) == profile]
        if not selected:
            print(f"\nNo rows found for profile: {profile}")
            continue

        out_dir = os.path.join(output_root, profile, "simulations")
        out_fields_dir = os.path.join(out_dir, "fields")
        os.makedirs(out_fields_dir, exist_ok=True)
        out_metadata = os.path.join(out_dir, "metadata.csv")

        missing_fields = []
        with open(out_metadata, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            for record in selected:
                clean_record = {col: record.get(col, "") for col in REQUIRED_COLUMNS}
                src = _source_field_path(clean_record, source_fields_dir)
                if src and os.path.exists(src):
                    dst = os.path.join(out_fields_dir, os.path.basename(src))
                    _copy_or_link(src, dst, args.link_fields)
                    clean_record["field_file"] = dst
                else:
                    missing_fields.append(clean_record.get("simulation_id") or f"line {record['_source_line_number']}")
                writer.writerow(clean_record)

        print(f"\nWrote {len(selected)} {profile} rows to:")
        print(f"- {out_metadata}")
        print(f"- {out_fields_dir}")
        if missing_fields:
            print(f"Warning: {len(missing_fields)} rows had missing field files.")
            print("First missing entries:")
            for item in missing_fields[:10]:
                print(f"- {item}")


def build_parser():
    parser = argparse.ArgumentParser(description="Split a mixed legacy/expanded TE-film database into clean folders.")
    parser.add_argument("--metadata", default=os.path.join("data", "simulations", "metadata.csv"))
    parser.add_argument("--source-fields", default=os.path.join("data", "simulations", "fields"))
    parser.add_argument("--output-root", default="split_databases")
    parser.add_argument("--profile", default="all", help="Profile to extract, e.g. expanded, legacy, or all.")
    parser.add_argument("--link-fields", action="store_true", help="Use hard links for HDF5 files when possible to avoid duplicating disk usage.")
    parser.add_argument("--dry-run", action="store_true", help="Only report detected profile counts.")
    return parser


if __name__ == "__main__":
    split_database(build_parser().parse_args())
