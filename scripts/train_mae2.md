# update the training workflow
- let the ssl_dataset, sft_dataset use different subset_frac so they can use different statisitcs
- intead interleaf the SSL and SFT every batches, do this: configure a `n_sft_epochs_per_ssl_epoch`, meaning, for each SSL epoch, do `n_sft_epochs_per_ssl_epoch` epochs of SFT, so that SFT could converge better to see the SSL feature impact
- in addition to current metrics, add a visulization for SSL results, make a png plot comparing the original un-masked image, the masked image that is input the backbone and the output image from the backbone.
- ask me if anything is unclear

# claude output
1. Separate subset fracs (ssl_subset_frac / sft_subset_frac)

Replaced single subset_frac with two independent params
Each dataset is subsampled independently so they can use different data volumes / class distributions
2. New training loop: n_sft_epochs_per_ssl_epoch

Removed n_ssl / n_sft (batches per interleave cycle) and the itertools.cycle pattern
Each SSL "epoch" is now one full pass through ssl_loader
After each SSL epoch, n_sft_epochs_per_ssl_epoch full passes through sft_loader are done
opt_sft and nu_flavor_head are created once and persist across SSL epochs (per your preference)
Per-SFT-epoch CE loss and accuracy are printed so you can watch convergence
3. SSL visualization

_sparse_to_dense: converts sparse (channel, tick) coords + features → 2D numpy array
_visualize_ssl: saves a 3-panel matplotlib PNG (original | masked | reconstructed) for the first event of the first valid batch
PNG saved to --viz_dir (default ./viz) as ssl_viz_epoch{NNNN}.png after every SSL epoch
Visualization runs in model.eval() + torch.no_grad() so it doesn't pollute gradients