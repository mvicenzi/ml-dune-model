import time
import psutil
import torch
from collections import defaultdict
from pathlib import Path
import json

class MetricsMonitor:
    """Lightweight training metrics tracker for comparing model architectures."""

    def __init__(self, model_name, save_dir="./metrics"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Metrics storage
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)

        # Timing
        self.epoch_start_time = None
        self.batch_start_time = None

        # GPU tracking
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None

        # CPU tracking
        self.process = psutil.Process()

        # Optional: sync CUDA before reading timing/memory (more accurate, slightly slower)
        self.sync_cuda = False

        # Internal: epoch slice index for batch metrics
        self._epoch_batch_start_idx = 0

        # ---------------------------
        # Peak tracking (what you asked for)
        # ---------------------------
        self._overall_gpu_peak_mb = 0.0            # peak GPU usage overall (training+validation)
        self._epoch_train_gpu_peak_mb = 0.0        # peak GPU usage during training within current epoch
        self._epoch_val_gpu_peak_mb = 0.0          # peak GPU usage during validation within current epoch

        # Helps guard against user forgetting to call validation hooks
        self._in_validation = False

    # ------------------ helpers ------------------

    def _maybe_sync(self):
        if self.device is not None and self.sync_cuda:
            torch.cuda.synchronize(self.device)

    def _read_gpu_peak_mb(self) -> float:
        """Read current CUDA 'max_memory_allocated' since last reset."""
        if self.device is None:
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device) / (1024**2))

    def _update_overall_peak(self, peak_mb: float):
        if peak_mb > self._overall_gpu_peak_mb:
            self._overall_gpu_peak_mb = peak_mb

    # ------------------ lifecycle ------------------

    def on_train_begin(self, model):
        """Call at training start to capture model info."""
        self.metrics["model_name"] = self.model_name
        self.metrics["total_params"] = sum(p.numel() for p in model.parameters())
        self.metrics["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.metrics["model_size_mb"] = (
            sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        )

        print(f"\n{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Total params: {self.metrics['total_params']:,}")
        print(f"Trainable params: {self.metrics['trainable_params']:,}")
        print(f"Model size: {self.metrics['model_size_mb']:.2f} MB")
        print(f"{'='*60}\n")

        # Reset CUDA peak stats for a clean baseline
        if self.device is not None:
            torch.cuda.reset_peak_memory_stats(self.device)

        # Reset overall peak tracker too
        self._overall_gpu_peak_mb = 0.0

    def on_epoch_begin(self, epoch):
        """Call at start of each epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

        # Mark where this epoch's batch metrics begin (robust slicing)
        self._epoch_batch_start_idx = len(self.metrics["batch_losses"])

        # Reset per-epoch trackers
        self._epoch_train_gpu_peak_mb = 0.0
        self._epoch_val_gpu_peak_mb = 0.0
        self._in_validation = False

        # Reset CUDA peak stats per epoch so "epoch peak" can be computed directly if desired
        if self.device is not None:
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_batch_begin(self):
        """Call at start of each training batch."""
        self.batch_start_time = time.time()

        # For per-batch peak usage, reset CUDA peak at batch start
        if self.device is not None:
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_batch_end(self, batch_idx, loss, batch_size):
        """Call after each training batch with loss value."""
        self._maybe_sync()
        batch_time = time.time() - self.batch_start_time

        # Core metrics
        self.metrics["batch_losses"].append(float(loss))
        self.metrics["batch_times"].append(batch_time)
        self.metrics["epochs_list"].append(self.current_epoch)
        self.metrics["batch_indices"].append(batch_idx)

        # Throughput
        samples_per_sec = batch_size / batch_time if batch_time > 0 else 0.0
        self.metrics["throughput"].append(samples_per_sec)

        # GPU memory metrics (allocated/reserved snapshots + per-batch peak)
        if self.device is not None:
            gpu_mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
            gpu_mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)

            # This is the *peak during the batch* because we reset at on_batch_begin().
            batch_gpu_peak_mb = self._read_gpu_peak_mb()

            self.metrics["gpu_memory_allocated_mb"].append(float(gpu_mem_allocated))
            self.metrics["gpu_memory_reserved_mb"].append(float(gpu_mem_reserved))

            # NEW: peak GPU per training batch
            self.metrics["gpu_memory_peak_batch_mb"].append(batch_gpu_peak_mb)

            # Track epoch training peak + overall peak
            if batch_gpu_peak_mb > self._epoch_train_gpu_peak_mb:
                self._epoch_train_gpu_peak_mb = batch_gpu_peak_mb
            self._update_overall_peak(batch_gpu_peak_mb)

        # CPU memory (RSS)
        cpu_mem_mb = self.process.memory_info().rss / (1024**2)
        self.metrics["cpu_memory_mb"].append(float(cpu_mem_mb))

    # ------------------ validation hooks (NEW) ------------------

    def on_validation_begin(self, epoch=None):
        """
        Call right before running validation for an epoch.
        (epoch is optional; included to be tolerant with your caller.)
        """
        self._in_validation = True
        if epoch is not None:
            self.current_epoch = epoch

        if self.device is not None:
            # Reset so "validation peak" is per-validation-phase
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_validation_end(self):
        """Call right after validation finishes for an epoch."""
        self._maybe_sync()

        if self.device is not None:
            val_peak_mb = self._read_gpu_peak_mb()

            # NEW: peak GPU during validation (per epoch)
            self.epoch_metrics["gpu_memory_peak_validation_mb"].append(float(val_peak_mb))

            # Update trackers
            if val_peak_mb > self._epoch_val_gpu_peak_mb:
                self._epoch_val_gpu_peak_mb = val_peak_mb
            self._update_overall_peak(val_peak_mb)
        else:
            # Keep list lengths consistent if running on CPU
            self.epoch_metrics["gpu_memory_peak_validation_mb"].append(0.0)

        self._in_validation = False

    def on_epoch_end(self, epoch, test_loss, test_acc):
        """Call at end of each epoch with test metrics (typically AFTER validation)."""
        self._maybe_sync()
        epoch_time = time.time() - self.epoch_start_time

        # Slice out this epoch's batches robustly
        epoch_start = self._epoch_batch_start_idx
        epoch_end = len(self.metrics["batch_losses"])
        epoch_losses = self.metrics["batch_losses"][epoch_start:epoch_end]

        self.epoch_metrics["epoch"].append(epoch)
        self.epoch_metrics["train_loss_mean"].append(sum(epoch_losses) / len(epoch_losses))
        self.epoch_metrics["train_loss_min"].append(min(epoch_losses))
        self.epoch_metrics["train_loss_max"].append(max(epoch_losses))
        self.epoch_metrics["test_loss"].append(float(test_loss))
        self.epoch_metrics["test_accuracy"].append(float(test_acc))
        self.epoch_metrics["epoch_time_sec"].append(epoch_time)

        # NEW: peak GPU during training in this epoch
        self.epoch_metrics["gpu_memory_peak_train_epoch_mb"].append(float(self._epoch_train_gpu_peak_mb))

        # NEW: peak GPU overall so far (training + validation)
        self.epoch_metrics["gpu_memory_peak_overall_mb"].append(float(self._overall_gpu_peak_mb))

        # Back-compat / additional GPU aggregates (optional but useful)
        if self.device is not None and epoch_end > epoch_start:
            epoch_gpu_alloc = self.metrics["gpu_memory_allocated_mb"][epoch_start:epoch_end]
            epoch_gpu_res = self.metrics["gpu_memory_reserved_mb"][epoch_start:epoch_end]
            epoch_gpu_batch_peaks = self.metrics["gpu_memory_peak_batch_mb"][epoch_start:epoch_end]

            self.epoch_metrics["gpu_memory_mean_allocated_mb"].append(
                sum(epoch_gpu_alloc) / len(epoch_gpu_alloc)
            )
            self.epoch_metrics["gpu_memory_mean_reserved_mb"].append(
                sum(epoch_gpu_res) / len(epoch_gpu_res)
            )

            # This is "max of per-batch peaks" which is typically what you want for train-phase peak.
            self.epoch_metrics["gpu_memory_peak_from_batches_mb"].append(max(epoch_gpu_batch_peaks))

        # CPU memory stats
        epoch_cpu_mem = self.metrics["cpu_memory_mb"][epoch_start:epoch_end]
        self.epoch_metrics["cpu_memory_peak_mb"].append(max(epoch_cpu_mem))

        # Throughput stats
        epoch_throughput = self.metrics["throughput"][epoch_start:epoch_end]
        self.epoch_metrics["throughput_mean"].append(sum(epoch_throughput) / len(epoch_throughput))

        print(
            f"Epoch {epoch} completed in {epoch_time:.2f}s | "
            f"Train Loss: {self.epoch_metrics['train_loss_mean'][-1]:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

        if self.device is not None:
            # Validation peak might not exist if user didn't call validation hooks.
            val_peak = None
            if len(self.epoch_metrics.get("gpu_memory_peak_validation_mb", [])) == len(self.epoch_metrics["epoch"]):
                val_peak = self.epoch_metrics["gpu_memory_peak_validation_mb"][-1]

            msg = (
                f"  GPU peaks: train-epoch {self.epoch_metrics['gpu_memory_peak_train_epoch_mb'][-1]:.1f} MB"
            )
            if val_peak is not None:
                msg += f" | val {val_peak:.1f} MB"
            msg += f" | overall {self.epoch_metrics['gpu_memory_peak_overall_mb'][-1]:.1f} MB"
            msg += f" | Throughput: {self.epoch_metrics['throughput_mean'][-1]:.1f} samples/sec"
            print(msg)

    def save(self):
        """Save metrics to JSON file."""
        output_path = self.save_dir / f"{self.model_name}_metrics.json"

        all_metrics = {
            "model_info": {
                "model_name": self.metrics["model_name"],
                "total_params": self.metrics["total_params"],
                "trainable_params": self.metrics["trainable_params"],
                "model_size_mb": self.metrics["model_size_mb"],
            },
            "batch_metrics": {
                k: v
                for k, v in self.metrics.items()
                if k not in ["model_name", "total_params", "trainable_params", "model_size_mb"]
            },
            "epoch_metrics": dict(self.epoch_metrics),
        }

        with open(output_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n✓ Metrics saved to {output_path}")
        return output_path

    def print_summary(self):
        """Print final summary statistics."""
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY - {self.model_name}")
        print(f"{'='*60}")
        print(f"Total epochs: {len(self.epoch_metrics['epoch'])}")
        print(f"Final test accuracy: {self.epoch_metrics['test_accuracy'][-1]:.2f}%")
        print(f"Best test accuracy: {max(self.epoch_metrics['test_accuracy']):.2f}%")
        print(f"Final test loss: {self.epoch_metrics['test_loss'][-1]:.4f}")
        print(f"Total training time: {sum(self.epoch_metrics['epoch_time_sec']):.2f}s")
        print(
            f"Avg epoch time: "
            f"{sum(self.epoch_metrics['epoch_time_sec'])/len(self.epoch_metrics['epoch_time_sec']):.2f}s"
        )

        if self.device is not None:
            overall = (
                self._overall_gpu_peak_mb
                if len(self.epoch_metrics.get("gpu_memory_peak_overall_mb", [])) == 0
                else max(self.epoch_metrics["gpu_memory_peak_overall_mb"])
            )
            print(f"Peak GPU memory overall: {overall:.1f} MB")

            if self.epoch_metrics.get("gpu_memory_peak_train_epoch_mb"):
                print(
                    f"Peak GPU memory (train epoch): "
                    f"{max(self.epoch_metrics['gpu_memory_peak_train_epoch_mb']):.1f} MB"
                )

            if self.epoch_metrics.get("gpu_memory_peak_validation_mb"):
                print(
                    f"Peak GPU memory (validation): "
                    f"{max(self.epoch_metrics['gpu_memory_peak_validation_mb']):.1f} MB"
                )

        print(f"Peak CPU memory: {max(self.epoch_metrics['cpu_memory_peak_mb']):.1f} MB")
        print(
            f"Avg throughput: "
            f"{sum(self.epoch_metrics['throughput_mean'])/len(self.epoch_metrics['throughput_mean']):.1f} samples/sec"
        )
        print(f"{'='*60}\n")
