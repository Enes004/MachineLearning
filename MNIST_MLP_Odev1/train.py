"""Train and evaluate an MLP on MNIST."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mlp_model import MNISTMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare three MLP sizes using validation accuracy; test only the winner.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use auto to select CUDA when available, otherwise CPU.",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive.")
    if len(args.hidden_dims) < 2 or any(width <= 0 for width in args.hidden_dims):
        parser.error("hidden-dims must contain at least two positive layer sizes.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but it is not available.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def make_optimizer(name: str, parameters: Any, learning_rate: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    return torch.optim.SGD(parameters, lr=learning_rate)


def measure(
    model: MNISTMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def train_configuration(
    hidden_dims: list[int],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model = MNISTMLP(hidden_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.learning_rate)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_validation_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)

        train_loss = total_loss / count
        train_accuracy = correct / count
        validation_loss, validation_accuracy = measure(
            model, validation_loader, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"  Epoch {epoch:02d}/{args.epochs}: "
            f"train loss={train_loss:.4f}, train acc={train_accuracy:.2%}, "
            f"val loss={validation_loss:.4f}, val acc={validation_accuracy:.2%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training finished without a validation checkpoint.")
    model.load_state_dict(best_state)
    return {
        "hidden_dims": hidden_dims,
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "train_loss": history["train_loss"][best_epoch - 1],
        "train_accuracy": history["train_accuracy"][best_epoch - 1],
        "validation_loss": history["validation_loss"][best_epoch - 1],
        "validation_accuracy": best_validation_accuracy,
        "training_seconds": time.perf_counter() - started,
    }


def save_curves(history: dict[str, list[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["validation_loss"], marker="o", label="Validation")
    axes[0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train")
    axes[1].plot(epochs, history["validation_accuracy"], marker="o", label="Validation")
    axes[1].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_comparison(experiments: list[dict[str, Any]], path: Path) -> None:
    labels = ["-".join(map(str, experiment["hidden_dims"])) for experiment in experiments]
    values = [experiment["validation_accuracy"] for experiment in experiments]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color="#3973ac")
    axis.set(title="Validation accuracy by MLP configuration", xlabel="Hidden-layer widths")
    axis.set_ylabel("Best validation accuracy")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_results_report(results: dict[str, Any], path: Path) -> None:
    experiment_rows = [
        f"| {', '.join(map(str, experiment['hidden_dims']))} "
        f"| {experiment['best_epoch']} "
        f"| {experiment['best_validation_accuracy']:.2%} "
        f"| {experiment['training_seconds']:.1f} |"
        for experiment in results["experiments"]
    ]
    report = [
        "# MNIST MLP Deney Sonuclari",
        "",
        "Bu rapor `train.py` tarafindan egitim sonunda olusturulmustur.",
        "El yazisi degerlendirmesini ve genel yorumunuzu `REPORT.md` dosyasina ekleyin.",
        "",
        "## Veri bolumu",
        "",
        "| Train | Validation | Test |",
        "|---:|---:|---:|",
        (
            f"| {results['dataset_split']['train']} "
            f"| {results['dataset_split']['validation']} "
            f"| {results['dataset_split']['test']} |"
        ),
        "",
        "Test kumesi yalnizca validation ile model secimi tamamlandiktan sonra "
        "degerlendirilmistir.",
        "",
        "## Hiperparametreler ve son olcumler",
        "",
        "| Parametre/olcum | Deger |",
        "|---|---:|",
        f"| Gizli katman boyutlari | {', '.join(map(str, results['selected_hidden_dims']))} |",
        "| Aktivasyon | ReLU |",
        "| Kayip | CrossEntropyLoss |",
        f"| Optimizer | {results['optimizer']} |",
        f"| Ogrenme orani | {results['learning_rate']} |",
        f"| Batch size | {results['batch_size']} |",
        f"| Epoch sayisi | {results['epochs']} |",
        f"| Seed | {results['seed']} |",
        f"| Cihaz | {results['device']} |",
        f"| En iyi validation epoch'u | {results['selected_epoch']} |",
        f"| Training loss (secilen epoch) | {results['train_loss_at_selected_epoch']:.4f} |",
        f"| Validation loss (secilen epoch) | {results['validation_loss_at_selected_epoch']:.4f} |",
        f"| Training accuracy (secilen epoch) | {results['train_accuracy_at_selected_epoch']:.2%} |",
        f"| Validation accuracy (secilen epoch) | {results['validation_accuracy_at_selected_epoch']:.2%} |",
        f"| Test accuracy | {results['test_accuracy']:.2%} |",
        f"| Secilen modelin egitim suresi | {results['selected_training_seconds']:.1f} saniye |",
        "",
        "## Yapilandirma karsilastirmasi",
        "",
        "| Gizli katman boyutlari | En iyi epoch | Validation accuracy | Egitim suresi (saniye) |",
        "|---|---:|---:|---:|",
        *experiment_rows,
        "",
        "## Grafikler",
        "",
        "![Train ve validation loss/accuracy](artifacts/training_curves.png)",
    ]
    if len(results["experiments"]) > 1:
        report.extend(
            (
                "",
                "![Yapilandirma karsilastirmasi](artifacts/configuration_comparison.png)",
            )
        )
    report.extend(
        (
            "",
            "El yazisi tahminlerini ve kendi degerlendirmenizi `REPORT.md` dosyasinda "
            "tamamlayin.",
            "",
        )
    )
    path.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print("Preparing MNIST: 55,000 train / 5,000 validation / 10,000 test.")

    transform = transforms.ToTensor()
    training_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_data, validation_data = random_split(
        training_data,
        (55_000, 5_000),
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=loader_generator
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    configurations = (
        [[128, 64], [256, 128], [512, 256, 128]]
        if args.compare
        else [args.hidden_dims]
    )
    experiments: list[dict[str, Any]] = []
    for hidden_dims in configurations:
        print(f"\nTraining hidden layers {hidden_dims}")
        experiments.append(
            train_configuration(hidden_dims, train_loader, validation_loader, args, device)
        )

    best = max(experiments, key=lambda experiment: experiment["validation_accuracy"])
    test_loss, test_accuracy = measure(
        best["model"], test_loader, nn.CrossEntropyLoss(), device
    )
    checkpoint_path = args.output_dir / "best_mnist_mlp.pt"
    torch.save(
        {
            "model_state_dict": best["model"].state_dict(),
            "hidden_dims": best["hidden_dims"],
            "normalization": "ToTensor: grayscale pixels scaled to [0, 1]",
            "seed": args.seed,
        },
        checkpoint_path,
    )

    save_curves(best["history"], args.output_dir / "training_curves.png")
    if args.compare:
        save_comparison(experiments, args.output_dir / "configuration_comparison.png")

    results = {
        "dataset_split": {"train": 55_000, "validation": 5_000, "test": 10_000},
        "selected_hidden_dims": best["hidden_dims"],
        "selected_epoch": best["best_epoch"],
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "training_seconds_all_configurations": sum(
            experiment["training_seconds"] for experiment in experiments
        ),
        "selected_training_seconds": best["training_seconds"],
        "train_loss_at_selected_epoch": best["train_loss"],
        "train_accuracy_at_selected_epoch": best["train_accuracy"],
        "validation_loss_at_selected_epoch": best["validation_loss"],
        "validation_accuracy_at_selected_epoch": best["validation_accuracy"],
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "history": best["history"],
        "experiments": [
            {
                "hidden_dims": experiment["hidden_dims"],
                "best_epoch": experiment["best_epoch"],
                "best_validation_accuracy": experiment["best_validation_accuracy"],
                "training_seconds": experiment["training_seconds"],
            }
            for experiment in experiments
        ],
    }
    results_path = args.output_dir / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_results_report(results, Path("RESULTS.md"))
    print("\nFinal evaluation (test set used only after selecting by validation accuracy):")
    print(
        f"Selected layers: {best['hidden_dims']} | "
        f"train loss: {best['train_loss']:.4f} | "
        f"validation loss: {best['validation_loss']:.4f} | "
        f"train accuracy: {best['train_accuracy']:.2%} | "
        f"validation accuracy: {best['validation_accuracy']:.2%} | "
        f"test accuracy: {test_accuracy:.2%}"
    )
    print(f"Training time (selected model): {best['training_seconds']:.1f} seconds")
    print(f"Saved metrics to {results_path}, report to RESULTS.md, and model to {checkpoint_path}")


if __name__ == "__main__":
    main()
