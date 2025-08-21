# run_data_efficiency.py

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import argparse
import wandb

# Import your models and the new trainer function
from model import QNN
from cnn_model import ClassicalCNN
from trainer import train_and_evaluate
import matplotlib.pyplot as plt


# --- Custom Dataset for Quantum Data (copied from your train.py) ---
class QuantumProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.data = torch.load(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    # --- 1. Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run data efficiency experiment for QNN vs Classical CNN."
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs for each run."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--percentages",
        nargs="+",
        type=int,
        default=[1, 5, 10, 25, 50, 75, 100],
        help="List of data percentages to test.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fmnist", "kmnist"],
        default="mnist",
        help="Dataset to use.",
    )
    args = parser.parse_args()

    # --- 2. Initialize a single W&B run for the whole experiment ---
    wandb.init(
        project="quanvolutional-nn-mnist",
        job_type="data-efficiency-experiment",
        config=args,
    )
    config = wandb.config

    # Create a custom table to log results for a final plot
    results_table = wandb.Table(
        columns=["Model Type", "Data Percentage", "Test Accuracy"]
    )

    # --- 3. Load Full Datasets Once ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Load Classical Data
    transform = transforms.Compose([transforms.ToTensor()])
    if config.dataset == "mnist":
        full_train_classical = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_classical = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif config.dataset == "fmnist":
        full_train_classical = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_classical = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:  # kmnist
        full_train_classical = datasets.KMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_classical = datasets.KMNIST(
            root="./data", train=False, download=True, transform=transform
        )

    # Load Quantum Data (assuming it's pre-processed and named systematically)
    # Example path: ./data/fmnist_1_layer_processed_train.pt
    # NOTE: You will need to adjust these paths based on your actual pre-processed file names.
    q_train_path = f"./data/processed_train_{config.dataset}.pt"
    q_test_path = f"./data/processed_test_{config.dataset}.pt"
    full_train_qnn = QuantumProcessedDataset(q_train_path)
    test_qnn = QuantumProcessedDataset(q_test_path)

    test_loader_classical = DataLoader(test_classical, batch_size=config.batch_size)
    test_loader_qnn = DataLoader(test_qnn, batch_size=config.batch_size)

    qnn_accuracies = []
    classical_accuracies = []

    # --- 4. Main Experiment Loop ---
    for p in config.percentages:
        print(f"\n===== TRAINING ON {p}% OF THE DATA =====\n")

        # --- Subset the data ---
        num_train_samples = int((p / 100) * len(full_train_classical))
        # Use random_split to get a random subset and a small validation set
        subset_train_classical, _ = random_split(
            full_train_classical,
            [num_train_samples, len(full_train_classical) - num_train_samples],
        )
        subset_train_qnn, _ = random_split(
            full_train_qnn, [num_train_samples, len(full_train_qnn) - num_train_samples]
        )

        # Create DataLoaders for the subsets. A small validation set is still useful.
        # For simplicity, we create a dummy val_loader if the subset is too small.
        # A more robust way is to ensure val_set has at least 1 sample.
        train_loader_classical = DataLoader(
            subset_train_classical, batch_size=config.batch_size, shuffle=True
        )
        train_loader_qnn = DataLoader(
            subset_train_qnn, batch_size=config.batch_size, shuffle=True
        )

        # --- Train and Evaluate Classical Model ---
        print(f"--- Training Classical CNN on {num_train_samples} samples ---")
        classical_model = ClassicalCNN()
        classical_acc = train_and_evaluate(
            model=classical_model,
            train_loader=train_loader_classical,
            test_loader=test_loader_classical,
            config=config,
            device=device,
        )
        wandb.log({"classical_accuracy": classical_acc, "data_percentage": p})
        results_table.add_data("Classical", p, classical_acc)
        classical_accuracies.append(classical_acc)  # --- NEW: Store result

        # --- Train and Evaluate QNN Model ---
        print(f"--- Training QNN on {num_train_samples} samples ---")
        qnn_model = QNN()
        qnn_acc = train_and_evaluate(
            model=qnn_model,
            train_loader=train_loader_qnn,
            test_loader=test_loader_qnn,
            config=config,
            device=device,
        )
        wandb.log({"qnn_accuracy": qnn_acc, "data_percentage": p})
        results_table.add_data("QNN", p, qnn_acc)
        qnn_accuracies.append(qnn_acc)

    # --- 5. Create, Save, and Log the Final Plot ---
    print("\n--- Generating and logging final plot ---")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(config.percentages, qnn_accuracies, "o-", label="Quanv-CNN", color="blue")
    ax.plot(
        config.percentages,
        classical_accuracies,
        "s-",
        label="Classical CNN",
        color="green",
    )

    # Adding plot details
    ax.set_title(f"Data Efficiency on {config.dataset.upper()}: QNN vs. Classical CNN")
    ax.set_xlabel("Percentage of Training Data Used (%)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_xticks(config.percentages)
    ax.set_ylim(bottom=min(min(qnn_accuracies), min(classical_accuracies)) - 5, top=100)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the plot locally
    plt.savefig(f"data_efficiency_{config.dataset}.png")

    # Log the plot and the results table to W&B
    wandb.log(
        {
            "data_efficiency_plot": wandb.Image(fig),
            "data_efficiency_results": results_table,
        }
    )

    plt.close(fig)  # Close the plot to free memory

    print("\n===== EXPERIMENT COMPLETE =====\n")
    wandb.finish()


if __name__ == "__main__":
    main()
