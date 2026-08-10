# Evaluation suite

Standard metrics for comparing productions, and the home for model evaluation:
this package supersedes the tools under `dino/diagnostics/`.

The backbone is frozen in the strongest sense: features are extracted once to
disk and never recomputed. What gets trained is a small readout head, fresh for
every (checkpoint × feature source × metric).

### 1. Extract the features — GPU, once per checkpoint

`extract_features.py` runs a frozen checkpoint over a dataset once and writes
per-pixel features, truth and provenance to `.npz`; each probe reads that file
and scores the backbone, so results are comparable across epochs, runs
and training objectives.

```bash
python -m probes.extract_features CKPT.pt --pixel_truth --device=cuda
```
Every metric reads the `.npz` this writes, so the choices that matter are fixed
here and cannot be revisited downstream:

- `--pixel_truth` — writes `pixel_labels`, the 7-class PID taxonomy.
- `--extra_truth` — energy fraction, track id and truth charge.
- `--max_images` — how many events, default 10000 (`-1` for the whole dataset). This is not a runtime knob: at 2000 the pools behind the rare PIDs came from too few events (Michel from 113 of 397 validation events), which sets the noise floor under every per-class average. Note that the sharded reader rounds down to whole batches, so 10000 at `--batch_size=32` extracts 9984. Strongly affects the size of the resulting .npz files.
- `--source` — `student` (default) or `teacher`, one branch per file.

### 2. Score the metrics — CPU, as often as you like

Probes import neither `warpconvnet` nor a dataset reader, so they are CPU-only
and cheap to re-run. One GPU extraction pass feeds every metric.

```bash
python -m probes.probe_pid FEATURES.npz --out pid_ep100.json
python -m probes.probe_knn_pid FEATURES.npz --out pixelknn_ep100.json
python -m probes.probe_overlap FEATURES.npz --out overlap_ep100.json
python -m probes.event_probe FEATURES.npz --out event_ep100.json
```

## Available metrics

| Metric | Module | What it asks |
|---|---|---|
| PID | `probe_pid.py` | can a trained head read a pixel's particle type off the frozen features? |
| kNN PID | `probe_knn_pid.py` | do a pixel's nearest neighbours in feature space already carry its class? |
| Overlap | `probe_overlap.py` | can a trained head tell that a pixel's charge is shared between particles? |
| Event flavor | `event_probe.py` | do whole events of the same interaction flavor land near each other once pooled? |


## PID

Procedure: train a small head (MLP or linear) to classify pixels into PID and compare with truth.

- Split samples in 80/20 for training: this split is at the event level to avoid leakage.
- Sample up to `--pool_per_class` pixels per true PID (default 10000) from the training events, and again from the validation events. This balancing is needed to avoid rare PIDs affecting the metrics.
- Normalize the features, fitted on the training pixels only.
- Train two heads with fixed hyperparameters: a linear SVM and a MLP with one hidden layer of 128 (Adam, 30 epochs, no early stopping).
- Score F1 per PID on the validation pixels, then averaged over the six particle types for "macro-F1". 
- Also score macro-IoU, purity and efficiency from the same confusion matrix. 
- Repeat all of the above on the raw charge inputs (`channel`, `tick` and charge): the difference is what the backbone added.

### Notes:
All of these come from the same confusion matrix.
Counting validation pixels one PID at a time: 
- TP = true positive (really this PID, predicted as this PID)
- FP = false positive (really another PID, predicted as this one)
- FN = false negative (really this PID, predicted as something else)

- Efficiency (recall) = of the pixels that really are this PID, how many were found, TP/(TP+FN). It does not depend on the pool's class proportions.
- Purity (precision) = of the pixels predicted as this PID, how many really are it, TP/(TP+FP).
- F1 = the harmonic mean of purity and efficiency, 2TP/(2TP+FP+FN), so it drops if either side is bad.
- IoU = intersection over union, TP/(TP+FP+FN), the same three counts as F1 but with the errors weighted twice as heavily. Always ≤ F1, which is why it is worth having: it separates checkpoints that F1 compresses.
- macro-X = the per-type values averaged with equal weight, so 'Michel' counts as much as 'Track'.
- Accuracy = the fraction of all pixels predicted correctly, ignoring type. Only meaningful next to the class proportions, which is why the macro numbers are the headline.

## k-NN PID

Procedure: without training anything, do k-NN clustering in cosine feature space and look for PID classes.

- Sample up to `--max_pixels_per_class` pixels per class (default 50000), capped per image by `--max_pixels_per_image` (default auto, which spreads each pool over ~200 events). It's important to cap per image to avoid using pixels from just a few events. 
- Unlike PID, background is excluded from this metric entirely rather than predicted-but-unscored.
- L2-normalize, then take a majority vote over the `--knn_k` (default 5) nearest neighbours by cosine similarity. No pixel is its own neighbour.
- Score overall and per-class accuracy; per-type and macro F1 are reported alongside for the purity side. Student and teacher are scored side by side, and PNGs are written unless `--no_plots`. 
- `--with_purity` adds neighbourhood label purity at several k, and `--plot_scatter` a 2-D UMAP/t-SNE view.

## Overlap

Procedure: train a head to answer one yes/no question about each pixel — is more than a fraction `t` of its charge someone else's? — and compare with truth.

- The target is `overlap = 1 - pixel_energyfrac`. `pixel_energyfrac` is the leading contributor's share of the pixel's energy, so `pixel_energyfrac = 1` gives `overlap = 0`, a pixel one particle owns outright, and two equal contributors give `overlap = 0.5`. Only pixels carrying truth take part; contamination is undefined without it.
- Split samples 80/20 at the event level, as PID does, so no event is on both sides.
- Sample a balanced training pool of `--train_per_class` pixels (default 20000) either side of the threshold. Balancing is on the training side only: it stops the head learning the prior, which is a training concern.
- Score on the natural population, subsampled uniformly to `--val_pixels` so the real proportion of contaminated pixels is kept. Unlike PID this is affordable here — contaminated is about 14% of truth pixels, not a fraction of a percent, so precision is a real number and not an artefact of the prior.
- Normalize the features, fitted on the training pixels only.
- Train the same two heads PID uses, with the same fixed hyperparameters: a linear SVM and an MLP with one hidden layer of 128.
- Score precision, efficiency and F1 for the contaminated class, against two degenerate answers measured on the same pixels: always answer "pure", and guess at random.
- Repeat all of the above on the raw charge inputs only (`channel`, `tick` and log charge): the difference is what the backbone added.
- Repeat at thresholds 0.1, 0.2 and 0.3, reporting each. The headline is 0.2; the sweep is there so a reader can see whether the answer depends on where the line was drawn. Each threshold is a different question, with its own proportion of contaminated pixels and its own chance level.
- Break the headline down per particle type, reporting each type's own contaminated rate beside its efficiency and F1.

### Notes:
Types are contaminated at very different rates — Blip 8.7%, DeltaRay 35.4% — so a type's efficiency only means something next to its own rate. That spread is also why the natural population is the one scored: every type is contaminated less than half the time, so a head that knows only the particle type scores F1 = 0 and can contribute nothing to the result. On a balanced pool it would score above zero, and some of the result would be PID in disguise.

## Event flavor

Procedure: without training anything, average each event's pixel features into a single vector and do k-NN in cosine space, looking for the interaction flavor (numuCC, nueCC, NC).

- Sample up to `--max_pixels_per_event` pixels per event (default 2000) and mean-pool them into one vector per event. Events smaller than the cap keep all their pixels, so only the large ones are thinned; both sides pool the same sampled pixels. Events with no pixels, or with an unknown flavor, are dropped and counted.
- No balancing: the natural flavor mix is scored as it comes, and recorded next to every number. Nothing is trained here, so there is no prior for a head to learn.
- L2-normalize, then take a majority vote over the k nearest events by cosine similarity, at k = 1, 5, 10, 20. No event is its own neighbour.
- Score accuracy, macro-F1 and neighbour purity, next to two degenerate answers measured on the same events: guessing uniformly, and always predicting the most common flavor.
- Repeat on the raw charge inputs only (`channel`, `tick` and log charge), mean-pooled the same way: the difference is what the backbone added, and it guards against a score that only reflects where an event sits or how much charge it deposited.

## Outputs

Result JSONs are keyed `<run>:<epoch tag>:<source>`, which is what lets
`compare.py` merge files from different metrics, epochs and trainings with no
bookkeeping. Each entry embeds its own provenance (checkpoint, epoch, backbone,
charge-transform parameters, extraction source) and the settings that produced it
(seed, pool size). Writes are incremental, so a long multi-file run stays
crash-safe.

