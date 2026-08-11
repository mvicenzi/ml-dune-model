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
python -m probes.probe_instance FEATURES.npz --out instance_ep100.json
python -m probes.probe_vertex FEATURES.npz --out vertex_ep100.json
python -m probes.probe_event FEATURES.npz --out event_ep100.json
```

## Available metrics

| Metric | Module | What it asks |
|---|---|---|
| PID | `probe_pid.py` | can a trained head read a pixel's particle type off the frozen features? |
| kNN PID | `probe_knn_pid.py` | do a pixel's nearest neighbours in feature space already carry its class? |
| Overlap | `probe_overlap.py` | can a trained head tell that a pixel's charge is shared between particles? |
| Instance | `probe_instance.py` | do a pixel's nearest neighbours belong to the same particle it does? |
| Vertex | `probe_vertex.py` | can a trained head tell that a pixel sits close to the interaction point? |
| Event flavor | `probe_event.py` | do whole events of the same interaction flavor land near each other once pooled? |


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

## Instance

Procedure: without training anything, look at each pixel's nearest neighbours inside its own event and predict which particle it belongs to, then compare with truth.

- Only pixels carrying instance truth take part. `pixel_trackid` is 0 where there is none, and taking its absolute value would merge every such pixel into one huge fake particle whose members all neighbour each other.
- Draw up to `--max_queries` pixels (default 100000) uniformly across all events, so each event contributes in proportion to its size.
- Within the pixel's own event, L2-normalize and take the `--knn_k` (default 5) nearest neighbours by cosine similarity. No pixel is its own neighbour. Neighbours are drawn from the whole event, not just the sampled pixels.
- The prediction is the most common particle among those neighbours. Score the fraction of pixels where that is the pixel's own particle.
- Repeat on the raw charge inputs only (`channel`, `tick` and log charge): the difference is what the backbone added.
- Repeat again on neighbours picked at random from the same event. This is chance, and it is not small: a particle holding half its event is the most common answer among random neighbours most of the time, so a big-particle score cannot be read without it.
- Report the score again for each size of the pixel's own particle (1, 2-3, 4-9, 10-99, 100-999, 1000+), each next to the fraction of pixels it holds.

### Notes:
The score is per pixel, not per particle. Half of all particles have three pixels or fewer, so averaging per particle would let particles holding 2% of the charge decide the number; it would also average over a different population than the one sampled. The cost is that big particles dominate — 85% of pixels belong to particles of 100+ — which is what the size breakdown is there to expose.

A pixel whose particle is a single pixel has no companion that could be voted for, so it is wrong whatever the features do. Those pixels are 0.7% of the total, and the ceiling reported beside each score is the fraction that could be right in principle.

There is no confusion matrix here, unlike PID. Particle ids are per-event labels, so id 7 in one event has nothing to do with id 7 in another and predictions cannot be pooled into a shared set of classes. Only the accuracy is poolable.

## Vertex

Procedure: train a head to answer one yes/no question about each pixel — is it within `r` pixels of the interaction point? — and compare with truth.

- The target is the distance from each pixel to the true vertex, measured in pixels. The event's 3D `vertex_xyz` is projected into this view's (channel, tick) with the wire geometry, and the distance is taken there, so the number is in the same units the pixels are stored in.
- The interaction vertex only, which is the one the containers store. About 5% of pixels are near it at the headline radius. `vertex_kind` records this, since a production labelling more vertices would be a denser and easier task.
- The projection needs one constant, the drift-to-tick offset `t0` (`--vertex_t0_ticks`, default -0.567). Events whose vertex lands outside the wire volume are dropped and counted rather than clamped to an edge.
- Every pixel takes part, whether or not it carries truth. Distance to the vertex is geometry: a noise pixel beside the vertex is still beside the vertex.
- Split samples 80/20 at the event level, as PID does. This matters more here than in the other probes: the distance varies smoothly across an event, so a pixel-level split would leave pixels of the same event on both sides and hand either head that event's answer.
- Sample a balanced training pool of `--train_per_class` pixels (default 50000) either side of the radius. Balancing is on the training side only, exactly as in Overlap.
- Score on the natural population, subsampled uniformly to `--val_pixels` (default 200000), so the real proportion of near-vertex pixels is kept — about 5% at the headline radius.
- Normalize the features, fitted on the training pixels only.
- Train one head: an MLP with one hidden layer of 128, the same fixed hyperparameters PID uses.
- Score precision, efficiency and F1 for the near class, next to a random guess measured on the same pixels.
- Repeat all of the above on the raw charge inputs only (`channel`, `tick` and log charge): the difference is what the backbone added.
- Repeat at radii 10, 20 and 30 pixels, reporting each. The headline is 20; the sweep is there so a reader can see whether the answer depends on where the line was drawn. Each radius is a different question, with its own proportion of near-vertex pixels and its own chance level.
- Report the number of near pixels actually scored at each radius, and flag when a radius could not fill its training pool. A tighter radius has fewer near pixels on both sides of the split, and the sweep is only comparable while every radius has enough of them.

### Notes:
The raw charge inputs are a strong baseline here by construction, more so than in the other probes: "near the vertex" is a statement about position, and `channel` and `tick` are position. A head given only those two can learn where vertices usually sit and score well without knowing anything about the event in front of it. This is still a fair comparison — the backbone is handed the same absolute coordinates, through the positional encoding at its bottleneck — but it means a negative feat-minus-raw difference says the features are a worse route to position than position itself, not that they are blind to the vertex.

The random guess is not there to be beaten — the raw charge inputs are what the features are measured against. It is there because both sides of that comparison are trained heads, and a head can collapse to a constant answer and still produce a difference that looks healthy. A collapsed head cannot clear the random guess, so it is what separates a weak score from a broken one.

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

