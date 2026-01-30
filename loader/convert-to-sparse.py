#!/usr/bin/env python3

import h5py
from pathlib import Path
import numpy as np  
import hdf5plugin

# -------------------------------
# Configuration
# -------------------------------

ROOTDIR = "/nfs/data/1/mvicenzi/apa-test-data/gzip2"          # change this to your HDF5 folder
OUTDIR  = "/nfs/data/1/mvicenzi/apa-test-data/sparse_gzip2"   # folder to save new sparse HDF5 files
#OUTDIR  = "/nfs/data/1/mvicenzi/apa-test-data/sparse_zstd5"   # folder to save new sparse HDF5 files


# names of datasets to convert to sparse 2D frames or save as 1D arrays
FRAMES_2D = ["frame_rebinned_reco", "frame_trackid_1st", "frame_pid_1st", "frame_trackid_2nd", "frame_pid_2nd"]
ARRAYS_1D = ["channels_rebinned_reco", "channels_trackid_1st", "channels_pid_1st", "channels_trackid_2nd", "channels_pid_2nd",
             "tickinfo_rebinned_reco", "tickinfo_trackid_1st", "tickinfo_pid_1st", "tickinfo_trackid_2nd", "tickinfo_pid_2nd"]

# compression configuration
COMPRESSION_TYPE = "gzip"
COMPRESSION_LEVEL = 2
#COMPRESSION_TYPE = hdf5plugin.Zstd(clevel=5)
#COMPRESSION_LEVEL = None


# -------------------------------
# Utilities
# -------------------------------

def dense_to_sparse_numpy(frame):
    
    # find coordinates of non-zero elements
    coords = np.nonzero(frame)
    coords = np.stack(coords, axis=1).astype(np.int32)
    # extract features at those coordinates
    features = frame[coords[:, 0], coords[:, 1]]
    return coords, features

def process_h5_file(inpath: Path, outpath: Path):
    """
    Load all selected groups in the HDF5, convert to sparse numpy tensors,
    save back into a new HDF5 file with required compression.
    """
    with h5py.File(inpath, "r") as fin, h5py.File(outpath, "w") as fout:
    
        for group in fin.keys():

            image_in_group = fin[group]
            image_out_group = fout.create_group(group)

            for name, ds in image_in_group.items():
                
                # if tagged for saving as a 2D frame:
                if name in FRAMES_2D:

                    frame = ds[()] # dense (H=hannels, W=ticks)
                    coords, features = dense_to_sparse_numpy(frame)

                    # save as datasets
                    subgroup = image_out_group.create_group(name)
                    subgroup.create_dataset(name="coords", data=coords, compression=COMPRESSION_TYPE, compression_opts=COMPRESSION_LEVEL)
                    subgroup.create_dataset(name="features", data=features, compression=COMPRESSION_TYPE, compression_opts=COMPRESSION_LEVEL)

                #if tagged for saving as 1D array:
                elif name in ARRAYS_1D:

                    arr = ds[()]  # dense 1D array
                    image_out_group.create_dataset(name=name, data=arr, compression=COMPRESSION_TYPE, compression_opts=COMPRESSION_LEVEL)

                else:
                    print(f"Skipping {inpath.name}:{group}/{name}")

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
        print(f"Saved sparse file: {outpath}\n")

# -------------------------------
# Entry point
# -------------------------------

if __name__ == "__main__":
    main(ROOTDIR, OUTDIR)

