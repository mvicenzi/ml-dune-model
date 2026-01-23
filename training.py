import fire                       # CLI helper: lets you run 'python file.py --arg=value'
import torch                      # Main PyTorch framework
import torch.optim as optim        # Optimizers (AdamW, SGD, etc.)
import torch.nn.functional as F    # Functional layer calls (stateless)
import warp as wp                  # NVIDIA Warp JIT backend (for GPU kernel acceleration)
from torch.optim.lr_scheduler import StepLR  # Reduces learning rate on schedule
from pathlib import Path

from models import MODEL_REGISTRY
from metrics_monitor import MetricsMonitor
from loader.dataset import DUNEImageDataset
from loader.splits import Subset, train_val_split

# ---------------------------------------------------------------------------
# Training and evaluation utilities
# ---------------------------------------------------------------------------

def train(model, device, train_loader, optimizer, epoch, monitor):
    """Single-epoch training loop."""
    model.train()
    monitor.on_epoch_begin(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        monitor.on_batch_begin()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)                      # Forward pass
        loss = F.nll_loss(output, target)         # Classification loss
        loss.backward()                           # Backpropagation
        optimizer.step()                          # Weight update

        monitor.on_batch_end(batch_idx, loss.item(), len(data))

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}")


def test(model, device, test_loader):
    """Evaluation loop (no gradient updates)."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Test: Avg loss={test_loss:.4f}, Acc={acc:.2f}%")
    return test_loss, acc


def main(
    model_name="attn_base",
    batch_size=30,
    test_batch_size=30,
    epochs=10,
    lr=1e-3,
    scheduler_step_size=10,
    gamma=0.7,
    device="cuda",
    metrics_dir="./metrics",
    checkpoints_dir="./checkpoints",
    test_mode=True
):
    """Main training driver."""
    wp.init()  # Initialize Warp backend
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)

    # Initialize metrics monitor
    monitor = MetricsMonitor(model_name, save_dir=metrics_dir)

    # --- Data loading ---
    # load DUNE CVN dataset from directories
    # currently only 3 classes: nu (NC), nue, nutau (actually numu)
    rootdir = "/nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd"
    label_tokens = [ "numu", "nue", "nutau", "NC" ]
    dataset = DUNEImageDataset(rootdir=rootdir, class_names=label_tokens, view_index=2, use_cache=True)

    ### for testing: small subset
    n_subset = 50000
    subset_indices = torch.randperm(len(dataset))[:n_subset]
    subset_ds = Subset(dataset, subset_indices)

    train_ds, val_ds = None, None
    if test_mode:
        print(f"Running in TEST_MODE with small subset of data {n_subset}.")
        train_ds, val_ds, _, _ = train_val_split(subset_ds, val_fraction=0.2, use_cache=False)
    else:
        train_ds, val_ds, _, _ = train_val_split(dataset, val_fraction=0.2, use_cache=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_ds, batch_size=test_batch_size, shuffle=True)

    # --- Model, optimizer, LR scheduler ---
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    ModelCls = MODEL_REGISTRY[model_name]
    model = ModelCls().to(device)

    monitor.on_train_begin(
        model,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        epochs=epochs,
        lr=lr,
        scheduler_step_size=scheduler_step_size,
        gamma=gamma
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, monitor)

        monitor.on_validation_begin(epoch)
        test_loss, acc = test(model, device, test_loader)
        monitor.on_validation_end()

        monitor.on_epoch_end(epoch, test_loss, acc)
        scheduler.step()

    # Save final model checkpoint
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(exist_ok=True)
    ckpt_path = checkpoints_dir / f"{model_name}_final.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved final model checkpoint to: {ckpt_path}")

    # Final summary and save
    monitor.print_summary()
    monitor.save()

if __name__ == "__main__":
    # Fire allows CLI usage, e.g.:
    #   python training.py --epochs=10 --lr=0.001 --device=cuda
    fire.Fire(main)
