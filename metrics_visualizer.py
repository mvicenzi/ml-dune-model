import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


class MetricsVisualizer:
    """Compare and visualize metrics across multiple model architectures."""

    def __init__(self, metrics_dir="./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.models_data = {}

    # ------------------ Loading helpers ------------------

    def load_all_metrics(self):
        """Load all metric files from the directory."""
        for file in self.metrics_dir.glob("*_metrics.json"):
            with open(file, "r") as f:
                data = json.load(f)
                model_name = data["model_info"]["model_name"]
                self.models_data[model_name] = data

        print(f"Loaded {len(self.models_data)} model runs: {list(self.models_data.keys())}")
        return self.models_data

    def _ensure_loaded(self) -> bool:
        """Make sure metrics are loaded; return False if nothing found."""
        if not self.models_data:
            self.load_all_metrics()
        if not self.models_data:
            print("No metrics files found!")
            return False
        return True

    def _iter_selected_models(self, model_names=None):
        """
        Yield (name, data) pairs.
        - If model_names is None: all models.
        - If model_names is a list: only those models.
        - Warns about any models not found.
        """
        if model_names is None:
            return self.models_data.items()

        if isinstance(model_names, str):
            model_names = [model_names]

        missing = [name for name in model_names if name not in self.models_data]
        if missing:
            print(
                f"Model(s) {missing} not found. "
                f"Available models: {list(self.models_data.keys())}"
            )

        return [(name, self.models_data[name]) for name in model_names if name in self.models_data]

    @staticmethod
    def _first_present(d: dict, keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    # ------------------ Individual plot functions ------------------

    def plot_training_loss(self, model_names=None, ax=None, save_path=None):
        """Training Loss Evolution."""
        if not self._ensure_loaded():
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        else:
            fig = None

        for name, data in self._iter_selected_models(model_names):
            losses = data["batch_metrics"]["batch_losses"]
            window = min(50, max(1, len(losses) // 10)) or 1
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax.plot(smoothed, label=name, alpha=0.8)

        ax.set_xlabel("Batch (smoothed)")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is not None and fig is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"Training loss plot saved to {save_path}")
        
        return ax

    def plot_test_accuracy(self, model_names=None, ax=None, save_path=None):
        """Test Accuracy by Epoch."""
        if not self._ensure_loaded():
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        else:
            fig = None

        for name, data in self._iter_selected_models(model_names):
            epochs = data["epoch_metrics"]["epoch"]
            acc = data["epoch_metrics"]["test_accuracy"]
            ax.plot(epochs, acc, marker="o", label=name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy [%]")
        ax.set_title("Test Accuracy Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is not None and fig is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"Test accuracy plot saved to {save_path}")
        
        return ax

    def plot_gpu_memory_peak_per_batch(self, model_names=None, ax=None, save_path=None):
        """
        GPU peak memory per TRAINING batch.
        New monitor key: batch_metrics['gpu_memory_peak_batch_mb']
        (Falls back to old 'gpu_memory_peak_mb' if present.)
        """
        if not self._ensure_loaded():
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
        else:
            fig = None

        any_plotted = False
        for name, data in self._iter_selected_models(model_names):
            bm = data.get("batch_metrics", {})
            mem = self._first_present(bm, ["gpu_memory_peak_batch_mb", "gpu_memory_peak_mb"], default=None)
            if mem is None:
                continue
            ax.plot(mem, label=name, alpha=0.5)
            any_plotted = True

        ax.set_xlabel("Batch")
        ax.set_ylabel("Peak GPU Memory [MB]")
        ax.set_title("GPU Peak Memory per Training Batch")
        if any_plotted:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is not None and fig is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"GPU memory per batch plot saved to {save_path}")
        
        return ax
    
    def plot_gpu_memory_peaks_by_epoch(self, model_names=None, ax=None, save_path=None):
        """
        GPU peak memory per epoch:
        - training peak
        - validation peak

        Styling:
        - Same color per model
        - Different markers for train / val
        - Legend:
            * one entry per model (color only)
            * one entry per marker meaning (black markers)
        """
        if not self._ensure_loaded():
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
        else:
            fig = None

        # Marker convention
        marker_map = {
            "train": "o",
            "val": "^",
        }

        # Keep handles to build two separate legends
        model_handles = []
        marker_handles = []

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, (name, data) in enumerate(self._iter_selected_models(model_names)):
            em = data.get("epoch_metrics", {})
            epochs = em.get("epoch", None)
            if not epochs:
                continue

            color = color_cycle[idx % len(color_cycle)]

            train_peak = self._first_present(
                em,
                ["gpu_memory_peak_train_epoch_mb", "gpu_memory_peak_from_batches_mb"],
                default=None,
            )
            val_peak = em.get("gpu_memory_peak_validation_mb", None)

            # Plot data
            if train_peak is not None:
                ax.plot(
                    epochs,
                    train_peak,
                    marker=marker_map["train"],
                    linestyle="-",
                    color=color,
                )
            if val_peak is not None:
                ax.plot(
                    epochs,
                    val_peak,
                    marker=marker_map["val"],
                    linestyle="--",
                    color=color,
                )

            # Dummy handle for model legend (color only)
            model_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=2,
                    label=name,
                )
            )

        # Marker legend (black markers)
        marker_handles = [
            plt.Line2D([0], [0], color="black", marker=marker_map["train"],
                    linestyle="None", label="Train epoch peak"),
            plt.Line2D([0], [0], color="black", marker=marker_map["val"],
                    linestyle="None", label="Validation peak"),
        ]

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Peak GPU Memory [MB]")
        ax.set_title("GPU Peak Memory by Epoch")

        ax.grid(True, alpha=0.3)

        # First legend: models (colors)
        legend_models = ax.legend(
            handles=model_handles,
            title="Model",
            loc="upper left",
            frameon=True,
        )

        # Second legend: marker meaning
        legend_markers = ax.legend(
            handles=marker_handles,
            title="Metric",
            loc="upper right",
            frameon=True,
        )

        # Add first legend back explicitly
        ax.add_artist(legend_models)
        
        if save_path is not None and fig is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"GPU memory by epoch plot saved to {save_path}")

        return ax


    def plot_throughput(self, model_names=None, ax=None, save_path=None):
        """Average throughput per model."""
        if not self._ensure_loaded():
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        else:
            fig = None

        names_list = []
        throughputs = []
        for name, data in self._iter_selected_models(model_names):
            names_list.append(name)
            avg_throughput = np.mean(data["epoch_metrics"]["throughput_mean"])
            throughputs.append(avg_throughput)

        ax.bar(names_list, throughputs)
        ax.set_ylabel("Samples per second [1/s]")
        ax.set_title("Average Throughput")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")
        
        if save_path is not None and fig is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"Throughput plot saved to {save_path}")
        
        return ax

    def plot_model_size_vs_accuracy(self, model_names=None, ax=None, save_path=None):
        """Model Size vs Final Accuracy."""
        if not self._ensure_loaded():
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        else:
            fig = None

        for name, data in self._iter_selected_models(model_names):
            params = data["model_info"]["total_params"] / 1e6
            final_acc = data["epoch_metrics"]["test_accuracy"][-1]
            ax.scatter(params, final_acc, s=100, label=name)
            ax.text(params, final_acc, f" {name}", fontsize=9)

        ax.set_xlabel("Parameters (millions)")
        ax.set_ylabel("Final Test Accuracy [%]")
        ax.set_title("Model Size vs Performance")
        ax.grid(True, alpha=0.3)
        
        if save_path is not None and fig is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"Model size vs accuracy plot saved to {save_path}")
        
        return ax

    def plot_training_time(self, model_names=None, ax=None, save_path=None):
        """Total training time per model."""
        if not self._ensure_loaded():
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        else:
            fig = None

        names_list = []
        total_times = []
        for name, data in self._iter_selected_models(model_names):
            names_list.append(name)
            total_time = sum(data["epoch_metrics"]["epoch_time_sec"])
            total_times.append(total_time)

        ax.bar(names_list, total_times)
        ax.set_ylabel("Total Training Time [s]")
        ax.set_title("Total Training Time")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")
        
        if save_path is not None and fig is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"Training time plot saved to {save_path}")
        
        return ax

    # ------------------ NEW: Memory-focused combined plots ------------------

    def plot_gpu_memory_stack(self, model_names=None, save_path=None):
        """
        One figure with 3 stacked GPU-memory plots:
          1) peak per training batch
          2) peak by epoch (train / val / overall)
          3) per-epoch peaks summarized (train vs val vs overall) as bars (max over epochs)

        This is meant to be readable when you compare many models.
        """
        if not self._ensure_loaded():
            return

        fig = plt.figure(figsize=(15, 10), dpi=200)

        ax1 = plt.subplot(3, 1, 1)
        self.plot_gpu_memory_peak_per_batch(model_names=model_names, ax=ax1)

        ax2 = plt.subplot(3, 1, 2)
        self.plot_gpu_memory_peaks_by_epoch(model_names=model_names, ax=ax2)

        ax3 = plt.subplot(3, 1, 3)
        self.plot_gpu_memory_epoch_peak_summary(model_names=model_names, ax=ax3)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"GPU memory stack plot saved to {save_path}")
        plt.show()
        plt.close()

    def plot_gpu_memory_epoch_peak_summary(self, model_names=None, ax=None, save_path=None):
        """
        Bar summary: for each model, plot max-over-epochs for:
          - train epoch peak
          - val peak
        """
        if not self._ensure_loaded():
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
        else:
            fig = None

        names = []
        train_max = []
        val_max = []

        for name, data in self._iter_selected_models(model_names):
            em = data.get("epoch_metrics", {})
            names.append(name)

            train = self._first_present(
                em,
                ["gpu_memory_peak_train_epoch_mb", "gpu_memory_peak_from_batches_mb"],
                default=None,
            )
            val = em.get("gpu_memory_peak_validation_mb", None)

            train_max.append(float(np.max(train)) if train is not None and len(train) else np.nan)
            val_max.append(float(np.max(val)) if val is not None and len(val) else np.nan)

        x = np.arange(len(names))
        width = 0.35

        ax.bar(x - width/2, train_max, width, label="Train epoch peak")
        ax.bar(x + width/2, val_max, width, label="Val peak")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Peak GPU Memory [MB]")
        ax.set_title("Peak GPU Memory Summary (max over epochs)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        
        if save_path is not None and fig is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"GPU memory peak summary plot saved to {save_path}")
        
        return ax

    def plot_all_memory_metrics(self, model_names=None, ax=None, save_path=None):
        """
        NEW: combines *all* memory metrics (GPU + CPU) into one plot.

        What it shows:
          - GPU allocated (per batch) [if present]
          - GPU reserved (per batch) [if present]
          - GPU peak per training batch (per batch) [if present]
          - CPU RSS (per batch) [if present]

        Notes:
          - Different scales -> uses a second y-axis for CPU.
          - This is a "shape" plot (time series), not a summary.
        """
        if not self._ensure_loaded():
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
        else:
            fig = None

        ax_cpu = ax.twinx()
        any_gpu = False
        any_cpu = False

        for name, data in self._iter_selected_models(model_names):
            bm = data.get("batch_metrics", {})

            gpu_alloc = bm.get("gpu_memory_allocated_mb", None)
            gpu_res = bm.get("gpu_memory_reserved_mb", None)
            gpu_peak = self._first_present(bm, ["gpu_memory_peak_batch_mb", "gpu_memory_peak_mb"], default=None)
            cpu_rss = bm.get("cpu_memory_mb", None)

            if gpu_alloc is not None:
                ax.plot(gpu_alloc, alpha=0.35, label=f"{name} gpu_alloc")
                any_gpu = True
            if gpu_res is not None:
                ax.plot(gpu_res, alpha=0.35, label=f"{name} gpu_reserved")
                any_gpu = True
            if gpu_peak is not None:
                ax.plot(gpu_peak, alpha=0.9, label=f"{name} gpu_peak_batch")
                any_gpu = True

            if cpu_rss is not None:
                ax_cpu.plot(cpu_rss, alpha=0.6, linestyle="--", label=f"{name} cpu_rss")
                any_cpu = True

        ax.set_xlabel("Batch")
        ax.set_ylabel("GPU Memory [MB]")
        ax_cpu.set_ylabel("CPU RSS [MB]")
        ax.set_title("All Memory Metrics (GPU allocated/reserved/peak + CPU RSS)")

        # Merge legends from both axes
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax_cpu.get_legend_handles_labels()
        if (any_gpu or any_cpu) and (handles1 or handles2):
            ax.legend(handles1 + handles2, labels1 + labels2, loc="best")

        ax.grid(True, alpha=0.3)
        
        if save_path is not None and fig is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")
            print(f"All memory metrics plot saved to {save_path}")
        
        return ax

    # ------------------ Combined comparison figure ------------------

    def plot_comparison(self, save_path="model_comparison.png", model_names=None):
        """
        2x3 grid:
          (1) train loss
          (2) test accuracy
          (3) GPU peak per training batch
          (4) GPU peaks per epoch (train/val/overall)
          (5) model size vs accuracy
          (6) training time
        """
        if not self._ensure_loaded():
            return

        fig = plt.figure(figsize=(16, 10), dpi=200)

        ax1 = plt.subplot(2, 3, 1)
        self.plot_training_loss(model_names=model_names, ax=ax1)

        ax2 = plt.subplot(2, 3, 2)
        self.plot_test_accuracy(model_names=model_names, ax=ax2)

        ax3 = plt.subplot(2, 3, 3)
        self.plot_gpu_memory_peak_per_batch(model_names=model_names, ax=ax3)

        ax4 = plt.subplot(2, 3, 4)
        self.plot_gpu_memory_peaks_by_epoch(model_names=model_names, ax=ax4)

        ax5 = plt.subplot(2, 3, 5)
        self.plot_model_size_vs_accuracy(model_names=model_names, ax=ax5)

        ax6 = plt.subplot(2, 3, 6)
        self.plot_training_time(model_names=model_names, ax=ax6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")
        plt.show()
        plt.close()

    # ------------------ Summary table (updated) ------------------

    def print_summary_table(self):
        """Print comparison table of key metrics."""
        if not self.models_data:
            self.load_all_metrics()

        print("\n" + "=" * 110)
        print(
            f"{'Model':<15} {'Params [M]':<12} {'Final Acc %':<12} {'Best Acc %':<12} "
            f"{'Time (s)':<12} {'Peak GPU TrainEp [MB]':<20} {'Peak GPU Val [MB]':<16}"
        )
        print("=" * 110)

        for model_name, data in self.models_data.items():
            params = data["model_info"]["total_params"] / 1e6
            final_acc = data["epoch_metrics"]["test_accuracy"][-1]
            best_acc = max(data["epoch_metrics"]["test_accuracy"])
            total_time = sum(data["epoch_metrics"]["epoch_time_sec"])

            em = data.get("epoch_metrics", {})

            peak_train = self._first_present(
                em,
                ["gpu_memory_peak_train_epoch_mb", "gpu_memory_peak_from_batches_mb"],
                default=None,
            )
            peak_val = self._first_present(em, ["gpu_memory_peak_validation_mb"], default=None)

            train_str = f"{max(peak_train):.1f}" if peak_train is not None else "N/A"
            val_str = f"{max(peak_val):.1f}" if peak_val is not None else "N/A"

            print(
                f"{model_name:<15} {params:<12.2f} {final_acc:<12.2f} {best_acc:<12.2f} "
                f"{total_time:<12.1f} {train_str:<20} {val_str:<16}"
            )

        print("=" * 110 + "\n")


if __name__ == "__main__":
    viz = MetricsVisualizer()
    viz.load_all_metrics()
    viz.print_summary_table()

    # Main comparison grid:
    viz.plot_comparison()

    # NEW: memory-only stacked figure (3 plots):
    # viz.plot_gpu_memory_stack(save_path="gpu_memory_stack.png")

    # NEW: combine all memory metrics (GPU alloc/res/peak + CPU RSS) into one plot:
    # viz.plot_all_memory_metrics()
