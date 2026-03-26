"""
Decompress .tgz files from a dataset directory into a training-ready layout.

Input layout:
    <input_root>/<...>/<name>.tgz
        └── sparse/
                └── *.h5

Output layout:
    <output_root>/<...>/<name>/
                    └── *.h5   (sparse/ prefix stripped)

Usage:
    python decompress.py --input_root /path/to/input --output_root /path/to/output
    python decompress.py --input_root /path/to/input --output_root /path/to/output --max_files 100
"""

import argparse
import tarfile
from pathlib import Path


def find_tgz_files(input_root: Path, max_files: int) -> list[Path]:
    tgz_files = sorted(input_root.rglob("*.tgz"))
    if max_files > 0:
        tgz_files = tgz_files[:max_files]
    return tgz_files


def decompress_tgz(tgz_path: Path, input_root: Path, output_root: Path) -> int:
    """Extract .h5 files from a single .tgz into the corresponding output directory.

    Returns the number of files extracted.
    """
    # Relative path from input root, e.g. 13717/1/001/out_monte-carlo-....tgz
    rel_path = tgz_path.relative_to(input_root)
    # Output dir: replace .tgz with a folder of the same stem
    out_dir = output_root / rel_path.parent / tgz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            member_path = Path(member.name)
            # Strip leading "sparse/" directory from archive paths
            parts = member_path.parts
            filename = Path(*parts[1:]) if len(parts) > 1 else member_path
            dest = out_dir / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted += 1

    return extracted


def main():
    parser = argparse.ArgumentParser(description="Decompress dataset .tgz files to training-ready layout.")
    parser.add_argument("--input_root", type=Path, required=True, help="Root directory containing .tgz files.")
    parser.add_argument("--output_root", type=Path, required=True, help="Root directory for extracted output.")
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Maximum number of .tgz files to process (0 = all).",
    )
    args = parser.parse_args()

    input_root: Path = args.input_root.resolve()
    output_root: Path = args.output_root.resolve()

    if not input_root.exists():
        parser.error(f"input_root does not exist: {input_root}")

    tgz_files = find_tgz_files(input_root, args.max_files)
    total = len(tgz_files)
    print(f"Found {total} .tgz file(s) to process.")

    for i, tgz_path in enumerate(tgz_files, 1):
        rel = tgz_path.relative_to(input_root)
        print(f"[{i}/{total}] {rel} ...", end=" ", flush=True)
        try:
            n = decompress_tgz(tgz_path, input_root, output_root)
            print(f"extracted {n} file(s).")
        except Exception as exc:
            print(f"ERROR: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
