1. update the SFT head in models/mae_model.py:

Current SFT head uses sparse_global_avg_pool, which I think is a bad choice, change it to something like a sparse CNN clasification type of network. Make it not too big or too small. I am targeting 10k - 100k samples in the final SFT.

2. update scripts/train_mae.md:

After each epoch of SSL, in addition to the SFT training, add another reference model that uses the mae_model.sft_head directly on the original un-masked image. Use the same configurations for that reference model. I want to compare the sft_head performance using the SSL feature and the original charge


3. Ask me if anything is not clear