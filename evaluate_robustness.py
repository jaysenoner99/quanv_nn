# evaluate_robustness.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import wandb
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Import your models
from model import QNN
from cnn_model import ClassicalCNN


def evaluate_model(model, test_loader, device):
    """Evaluates a model's accuracy on a test set."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness on noisy datasets."
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fmnist", "kmnist", "cifar10"],
        required=True,
    )
    parser.add_argument(
        "--gaussian_stds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    parser.add_argument(
        "--salt_pepper_amounts",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.15, 0.2, 0.25],
    )
    args = parser.parse_args()

    # --- 2. W&B Initialization ---
    wandb.init(
        project="quanvolutional-nn-mnist",
        job_type="noise-robustness-evaluation",
        config=args,
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 3. Load Pre-trained Models ---
    model_dir = "./saved_models"
    classical_model_path = os.path.join(model_dir, f"{config.dataset}_cnn.pt")
    qnn_model_path = os.path.join(model_dir, f"{config.dataset}_qnn.pt")

    print("Loading pre-trained models...")
    classical_model = ClassicalCNN()
    classical_model.load_state_dict(torch.load(classical_model_path))

    qnn_model = QNN()
    qnn_model.load_state_dict(torch.load(qnn_model_path))

    # --- 4. Noise Evaluation Loop ---
    results = {"gaussian": {}, "salt_pepper": {}}
    noisy_data_dir = "./data/noisy_test_sets"

    # --- GAUSSIAN NOISE ---
    print("\n--- Evaluating Robustness to Gaussian Noise ---")
    classical_accs, qnn_accs = [], []
    for std in tqdm(config.gaussian_stds, desc="Gaussian Noise Levels"):
        # Load datasets
        classical_path = os.path.join(
            noisy_data_dir, f"classical_{config.dataset}_gaussian_{std}.pt"
        )
        quanv_path = os.path.join(
            noisy_data_dir, f"quanv_{config.dataset}_gaussian_{std}.pt"
        )

        classical_images, classical_labels = torch.load(classical_path)
        quanv_images, quanv_labels = torch.load(quanv_path)

        classical_loader = DataLoader(
            TensorDataset(classical_images, classical_labels),
            batch_size=config.batch_size,
        )
        quanv_loader = DataLoader(
            TensorDataset(quanv_images, quanv_labels), batch_size=config.batch_size
        )

        # Evaluate models
        classical_acc = evaluate_model(classical_model, classical_loader, device)
        qnn_acc = evaluate_model(qnn_model, quanv_loader, device)
        classical_accs.append(classical_acc)
        qnn_accs.append(qnn_acc)
    results["gaussian"]["classical"] = classical_accs
    results["gaussian"]["qnn"] = qnn_accs

    # --- SALT & PEPPER NOISE ---
    print("\n--- Evaluating Robustness to Salt & Pepper Noise ---")
    classical_accs, qnn_accs = [], []
    for amount in tqdm(config.salt_pepper_amounts, desc="S&P Noise Levels"):
        classical_path = os.path.join(
            noisy_data_dir, f"classical_{config.dataset}_salt_pepper_{amount}.pt"
        )
        quanv_path = os.path.join(
            noisy_data_dir, f"quanv_{config.dataset}_salt_pepper_{amount}.pt"
        )

        classical_images, classical_labels = torch.load(classical_path)
        quanv_images, quanv_labels = torch.load(quanv_path)

        classical_loader = DataLoader(
            TensorDataset(classical_images, classical_labels),
            batch_size=config.batch_size,
        )
        quanv_loader = DataLoader(
            TensorDataset(quanv_images, quanv_labels), batch_size=config.batch_size
        )

        classical_acc = evaluate_model(classical_model, classical_loader, device)
        qnn_acc = evaluate_model(qnn_model, quanv_loader, device)
        classical_accs.append(classical_acc)
        qnn_accs.append(qnn_acc)
    results["salt_pepper"]["classical"] = classical_accs
    results["salt_pepper"]["qnn"] = qnn_accs

    print("Generating Gaussian noise plot...")
    fig_gaussian, ax_gaussian = plt.subplots(figsize=(8, 6))
    ax_gaussian.plot(
        config.gaussian_stds, results["gaussian"]["qnn"], "o-", label="Quanv-CNN"
    )
    ax_gaussian.plot(
        config.gaussian_stds,
        results["gaussian"]["classical"],
        "s-",
        label="Classical CNN",
    )
    ax_gaussian.set_title(f"Robustness to Gaussian Noise on {config.dataset.upper()}")
    ax_gaussian.set_xlabel("Noise Standard Deviation")
    ax_gaussian.set_ylabel("Test Accuracy (%)")
    ax_gaussian.legend()
    ax_gaussian.grid(True)
    plt.tight_layout()
    plt.savefig(f"./images/gaussian_robustness_{config.dataset}.png")

    # Log the Gaussian plot to W&B
    wandb.log({"gaussian_robustness_plot": wandb.Image(fig_gaussian)})
    plt.close(fig_gaussian)  # Close the figure to free up memory

    # --- Salt & Pepper Plot ---
    print("Generating Salt & Pepper noise plot...")
    fig_sp, ax_sp = plt.subplots(figsize=(8, 6))
    ax_sp.plot(
        config.salt_pepper_amounts,
        results["salt_pepper"]["qnn"],
        "o-",
        label="Quanv-CNN",
    )
    ax_sp.plot(
        config.salt_pepper_amounts,
        results["salt_pepper"]["classical"],
        "s-",
        label="Classical CNN",
    )
    ax_sp.set_title(f"Robustness to Salt & Pepper Noise on {config.dataset.upper()}")
    ax_sp.set_xlabel("Proportion of Noisy Pixels")
    ax_sp.set_ylabel("Test Accuracy (%)")
    ax_sp.legend()
    ax_sp.grid(True)
    plt.tight_layout()
    plt.savefig(f"./images/salt_pepper_robustness_{config.dataset}.png")

    # Log the Salt & Pepper plot to W&B
    wandb.log({"salt_pepper_robustness_plot": wandb.Image(fig_sp)})
    plt.close(fig_sp)
    print("\n===== EVALUATION COMPLETE =====\n")
    wandb.finish()


if __name__ == "__main__":
    main()
