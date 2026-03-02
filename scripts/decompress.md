# Usage:

## Process only first 50 files (for testing)
```bash
python scripts/decompress.py \
    --input_root /nfs/data/1/jjo/data/fdhd_sparse_training \
    --output_root /nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27-sample \
    --max_files 20
```
## Process all files
```bash
python scripts/decompress.py \
    --input_root /nfs/data/1/jjo/data/fdhd_sparse_training \
    --output_root /nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27
```

# Task: write python script in "scripts/decompress.py" to decompress data to be training ready maintaining the original folder structure.
## Description
- I have a dataset from production here: /nfs/data/1/jjo/data/fdhd_sparse_training
- It has a some folder structure with ".tgz" files in the end.
- Each .tgz file contains a folder "sparse"
- in the "sparse" folder, it has many ".h5" files.
- Now I want to decompress the tgz files to the same folder structure with h5 files in the end.
- e.g., 
`<input_root>/13717/1/001/out_monte-carlo-013717-000363_304869_189_1_20260224T033825Z.tgz` should be in here:
`<output_root>/13717/1/001/out_monte-carlo-013717-000363_304869_189_1_20260224T033825Z/*.h5`. Output like this:
```bash
.
└── 13717
    └── 1
        └── 001
            └── out_monte-carlo-013717-000363_304869_189_1_20260224T033825Z
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_metadata.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode0.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode10.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode11.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode1.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode2.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode3.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode4.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode5.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode6.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode7.h5
                ├── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode8.h5
                └── monte-carlo-013717-000363_304869_189_1_20260224T033825Z_pixeldata-anode9.h5
```
- input and output root should be configurable.
- add an option to process max_number_files. default to 0 to process all the tgz files.