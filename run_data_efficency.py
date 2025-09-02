# run_data_efficiency.py

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import argparse
import wandb
import matplotlib.pyplot as plt

# Import your models and the new trainer function
from model import QNN
from cnn_model import ClassicalCNN
from trainer import train_and_evaluate


# --- Custom Dataset for Quantum Data ---
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

    # --- 2. Initialize a single W&B run ---
    wandb.init(
        project="quanvolutional-nn-mnist",
        job_type="data-efficiency-experiment",
        config=args,
    )
    config = wandb.config

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
    elif config.dataset == "kmnist":  # kmnist
        full_train_classical = datasets.KMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_classical = datasets.KMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif config.dataset == "cifar10":
        cifar_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),  # RGB → grayscale
                transforms.Resize(28),  # resize shortest side to 28
                transforms.CenterCrop(28),  # crop to exactly 28x28
                transforms.ToTensor(),
            ]
        )

        full_train_classical = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=cifar_transform
        )
        test_classical = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=cifar_transform
        )
    # Load Quantum Data (1-Layer)
    q_train_path_1l = f"./data/processed_train_{config.dataset}.pt"
    q_test_path_1l = f"./data/processed_test_{config.dataset}.pt"
    full_train_qnn_1l = QuantumProcessedDataset(q_train_path_1l)
    test_qnn_1l = QuantumProcessedDataset(q_test_path_1l)

    # --- NEW: Load Quantum Data (2-Layer) ---
    q_train_path_2l = f"./data/processed_train_{config.dataset}_2l.pt"
    q_test_path_2l = f"./data/processed_test_{config.dataset}_2l.pt"
    full_train_qnn_2l = QuantumProcessedDataset(q_train_path_2l)
    test_qnn_2l = QuantumProcessedDataset(q_test_path_2l)

    # Create test loaders
    test_loader_classical = DataLoader(test_classical, batch_size=config.batch_size)
    test_loader_qnn_1l = DataLoader(test_qnn_1l, batch_size=config.batch_size)
    test_loader_qnn_2l = DataLoader(
        test_qnn_2l, batch_size=config.batch_size
    )  # --- NEW

    # Lists to store results for plotting
    qnn_1l_accuracies = []
    qnn_2l_accuracies = []  # --- NEW
    classical_accuracies = []

    # --- 4. Main Experiment Loop ---
    for p in config.percentages:
        print(f"\n===== TRAINING ON {p}% OF THE DATA =====\n")

        num_train_samples = int((p / 100) * len(full_train_classical))

        # Create subsets for all three data types
        subset_train_classical, _ = random_split(
            full_train_classical,
            [num_train_samples, len(full_train_classical) - num_train_samples],
        )
        subset_train_qnn_1l, _ = random_split(
            full_train_qnn_1l,
            [num_train_samples, len(full_train_qnn_1l) - num_train_samples],
        )
        subset_train_qnn_2l, _ = random_split(  # --- NEW
            full_train_qnn_2l,
            [num_train_samples, len(full_train_qnn_2l) - num_train_samples],
        )

        # Create DataLoaders for the subsets
        train_loader_classical = DataLoader(
            subset_train_classical, batch_size=config.batch_size, shuffle=True
        )
        train_loader_qnn_1l = DataLoader(
            subset_train_qnn_1l, batch_size=config.batch_size, shuffle=True
        )
        train_loader_qnn_2l = DataLoader(  # --- NEW
            subset_train_qnn_2l, batch_size=config.batch_size, shuffle=True
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
        classical_accuracies.append(classical_acc)

        # --- Train and Evaluate QNN Model (1-Layer) ---
        print(f"--- Training QNN (1-Layer) on {num_train_samples} samples ---")
        qnn_model_1l = QNN()
        qnn_1l_acc = train_and_evaluate(
            model=qnn_model_1l,
            train_loader=train_loader_qnn_1l,
            test_loader=test_loader_qnn_1l,
            config=config,
            device=device,
        )
        wandb.log({"qnn_1_layer_accuracy": qnn_1l_acc, "data_percentage": p})
        results_table.add_data("QNN (1 Layer)", p, qnn_1l_acc)
        qnn_1l_accuracies.append(qnn_1l_acc)

        # --- NEW: Train and Evaluate QNN Model (2-Layer) ---
        print(f"--- Training QNN (2-Layer) on {num_train_samples} samples ---")
        qnn_model_2l = QNN()  # The model architecture is the same
        qnn_2l_acc = train_and_evaluate(
            model=qnn_model_2l,
            train_loader=train_loader_qnn_2l,
            test_loader=test_loader_qnn_2l,
            config=config,
            device=device,
        )
        wandb.log({"qnn_2_layer_accuracy": qnn_2l_acc, "data_percentage": p})
        results_table.add_data("QNN (2 Layers)", p, qnn_2l_acc)
        qnn_2l_accuracies.append(qnn_2l_acc)

    # --- 5. Create, Save, and Log the Final Plot ---
    print("\n--- Generating and logging final plot ---")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        config.percentages,
        qnn_1l_accuracies,
        "o-",
        label="Quanv-CNN (1 Layer)",
        color="blue",
    )
    ax.plot(
        config.percentages,
        qnn_2l_accuracies,
        "^-",
        label="Quanv-CNN (2 Layers)",
        color="red",
    )  # --- NEW
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
    # --- NEW: Update ylim to include all three models ---
    all_accuracies = qnn_1l_accuracies + qnn_2l_accuracies + classical_accuracies
    ax.set_ylim(bottom=min(all_accuracies) - 5, top=100)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the plot locally
    plt.savefig(f"./images/data_efficiency_{config.dataset}_with_2l.png")

    # Log the plot and the results table to W&B
    wandb.log(
        {
            "data_efficiency_plot": wandb.Image(fig),
            "data_efficiency_results": results_table,
        }
    )

    plt.close(fig)

    print("\n===== EXPERIMENT COMPLETE =====\n")
    wandb.finish()


if __name__ == "__main__":
    main()
