#!/usr/bin/env python3

import h5py
from pathlib import Path
import hdf5plugin

# -------------------------------
# Configuration
# -------------------------------

ROOTDIR = "/nfs/data/1/mvicenzi/apa-test-data/gzip2"          # change this to your HDF5 folder
OUTDIR  = "/nfs/data/1/mvicenzi/apa-test-data/zstd5"      # folder to save compressed HDF5 files

# compression configuration
COMPRESSION_TYPE = hdf5plugin.Zstd(clevel=5)
COMPRESSION_LEVEL = None

# -------------------------------
# Utilities
# -------------------------------

def process_h5_file(inpath: Path, outpath: Path):
    """
    Copy all datasets in the HDF5 file to a new file using Zstd compression.
    No sparsification applied.
    """
    with h5py.File(inpath, "r") as fin, h5py.File(outpath, "w") as fout:
        for group in fin.keys():
            image_in_group = fin[group]
            image_out_group = fout.create_group(group)

            for name, ds in image_in_group.items():
                data = ds[()]  # load full dataset
                image_out_group.create_dataset(
                    name=name,
                    data=data,
                    compression=COMPRESSION_TYPE,
                    compression_opts=COMPRESSION_LEVEL
                )

# -------------------------------
# Main Loop
# -------------------------------

def main(rootdir, outdir):
    rootdir = Path(rootdir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Find all HDF5 files ending with "anode<APA>.h5"
    h5_files = list(rootdir.rglob("*anode*.h5"))
    print(f"Found {len(h5_files)} HDF5 files to process.")

    for h5_file in h5_files:
        print(f"Processing {h5_file.name}")

        # preserve subdirectory structure
        rel_path = h5_file.relative_to(rootdir).with_suffix(".h5")
        outpath = outdir / rel_path
        outpath.parent.mkdir(parents=True, exist_ok=True)

        process_h5_file(h5_file, outpath)
        print(f"Saved compressed file: {outpath}\n")

# -------------------------------
# Entry point
# -------------------------------

if __name__ == "__main__":
    main(ROOTDIR, OUTDIR)
