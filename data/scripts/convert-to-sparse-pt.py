#!/usr/bin/env python3
"""
Convert all HDF5 files in a directory into sparse pickle files using WarpConvNet Voxels.

Each HDF5 file may contain multiple groups with `frame_rebinned_reco`.
For each group, a sparse representation is stored in a dictionary:
    {group: {"coords": ..., "features": ..., "offsets": ...}}

Saved file: same name as HDF5, extension .pt
"""

import h5py
from pathlib import Path
import torch
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures

# -------------------------------
# Configuration
# -------------------------------

ROOTDIR = "/nfs/data/1/mvicenzi/apa-test-data/gzip2"       # change this to your HDF5 folder
OUTDIR  = "/nfs/data/1/mvicenzi/apa-test-data/pickle"      # folder to save .pt files
FRAMES_2D = ["frame_rebinned_reco", "frame_trackid_1st", "frame_pid_1st", "frame_trackid_2nd", "frame_pid_2nd"]
ARRAYS_1D = ["channels_rebinned_reco", "channels_trackid_1st", "channels_pid_1st", "channels_trackid_2nd", "channels_pid_2nd",
             "tickinfo_rebinned_reco", "tickinfo_trackid_1st", "tickinfo_pid_1st", "tickinfo_trackid_2nd", "tickinfo_pid_2nd"]
PICKLE_PROTOCOL = 5

# -------------------------------
# Utilities
# -------------------------------

def process_h5_file(h5_path: Path):
    """
    Load all groups in the HDF5, convert to sparse Voxels, return a dict.
    """
    sparse_groups = {}

    with h5py.File(h5_path, "r") as f:
        for group in f.keys():

            group_dict = {}
            for name, ds in f[group].items():
                
                # if tagged for saving as a 2D frame:
                if name in FRAMES_2D:

                    frame = ds[()] # dense (H=hannels, W=ticks)

                    # reshape into (B=1, C=1, H, W)
                    x = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0) 

                    # use WarpConvNet to convert dense to sparse
                    # NOTE: this ignores "zeros", possible problem if trackid=0 is valid?
                    vox = Voxels.from_dense(x)

                    group_dict[name] = {
                        "coords": vox.coordinate_tensor.cpu(),
                        "features": vox.feature_tensor.cpu(),
                        "offsets": vox.offsets.cpu(),
                    }

                #if tagged for saving as 1D array:
                elif name in ARRAYS_1D:

                    arr = ds[()]  # dense 1D array
                    group_dict[name] = torch.from_numpy(arr)

                else:
                    print(f"Skipping {h5_path.name}:{group}/{name}")

            if group_dict:
                sparse_groups[group] = group_dict

    return sparse_groups

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
        sparse_groups = process_h5_file(h5_file)

        # preserve subdirectory structure
        rel_path = h5_file.relative_to(rootdir).with_suffix(".pt")
        outpath = outdir / rel_path
        outpath.parent.mkdir(parents=True, exist_ok=True)

        torch.save(sparse_groups, outpath, pickle_protocol=PICKLE_PROTOCOL)
        print(f"Saved sparse file: {outpath}\n")

# -------------------------------
# Entry point
# -------------------------------

if __name__ == "__main__":
    main(ROOTDIR, OUTDIR)

