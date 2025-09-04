# run_tsne_visualization.py

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
import os

# Import your models
from model import QNN
from cnn_model import ClassicalCNN


# Custom Dataset for Quantum Data
class QuantumProcessedDataset(Dataset):
    def __init__(self, filepath):
        self.data = torch.load(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_features(model, loader, device):
    """Extracts features and labels from a model and data loader."""
    model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(
            loader, desc=f"Extracting features from {model.__class__.__name__}"
        ):
            images = images.to(device)
            features = model.extract_features(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Run t-SNE feature space visualization."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fmnist", "kmnist", "cifar10"],
        required=True,
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for feature extraction."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to use for t-SNE (it can be slow).",
    )
    args = parser.parse_args()

    wandb.init(
        project="quanvolutional-nn-mnist", job_type="tsne-visualization", config=args
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # --- 1. Load Pre-trained Models ---
    model_dir = "./saved_models"
    classical_model_path = os.path.join(model_dir, f"{config.dataset}_cnn.pt")
    qnn_model_path = os.path.join(model_dir, f"{config.dataset}_qnn.pt")

    print("Loading pre-trained models...")
    classical_model = ClassicalCNN()
    classical_model.load_state_dict(torch.load(classical_model_path))
    qnn_model = QNN()
    qnn_model.load_state_dict(torch.load(qnn_model_path))

    # --- 2. Load Datasets ---
    # Classical Test Set
    transform = transforms.Compose([transforms.ToTensor()])
    if config.dataset == "mnist":
        classical_test_set = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif config.dataset == "fmnist":
        classical_test_set = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif config.dataset == "kmnist":
        classical_test_set = datasets.KMNIST(
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
        classical_test_set = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=cifar_transform
        )

    # Quantum-Processed Test Set
    q_test_path = f"./data/processed_test_{config.dataset}.pt"
    qnn_test_set = QuantumProcessedDataset(q_test_path)

    # Create DataLoaders
    classical_loader = DataLoader(
        classical_test_set, batch_size=config.batch_size, shuffle=False
    )
    qnn_loader = DataLoader(qnn_test_set, batch_size=config.batch_size, shuffle=False)

    # --- 3. Extract Features ---
    classical_features, classical_labels = get_features(
        classical_model, classical_loader, device
    )
    qnn_features, _ = get_features(qnn_model, qnn_loader, device)

    # --- 4. Subsample and Run t-SNE ---
    # Using a subset of data because t-SNE is computationally expensive
    indices = np.random.choice(
        classical_features.shape[0], config.num_samples, replace=False
    )
    print(f"Running t-SNE on {config.num_samples} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)

    classical_tsne = tsne.fit_transform(classical_features[indices])
    # Re-initialize for a fair comparison, though not strictly necessary
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    qnn_tsne = tsne.fit_transform(qnn_features[indices])

    labels_subset = classical_labels[indices]

    # --- 5. Plot the Results ---
    print("Generating plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot for Classical CNN
    scatter1 = ax1.scatter(
        classical_tsne[:, 0],
        classical_tsne[:, 1],
        c=labels_subset,
        cmap="tab10",
        alpha=0.7,
    )
    ax1.set_title("Classical CNN Feature Space")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")
    ax1.legend(handles=scatter1.legend_elements()[0], labels=list(range(10)))

    # Plot for Quanv-CNN
    scatter2 = ax2.scatter(
        qnn_tsne[:, 0], qnn_tsne[:, 1], c=labels_subset, cmap="tab10", alpha=0.7
    )
    ax2.set_title("Quanv-CNN Feature Space")
    ax2.set_xlabel("t-SNE Component 1")
    ax2.set_ylabel("t-SNE Component 2")
    ax2.legend(handles=scatter2.legend_elements()[0], labels=list(range(10)))

    fig.suptitle(
        f"t-SNE Visualization of Feature Spaces on {config.dataset.upper()} Test Set",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 6. Log to W&B ---
    plt.savefig(f"./images/tsne_visualization_{config.dataset}.png")
    wandb.log({"tsne_feature_space_plot": wandb.Image(fig)})
    plt.close(fig)

    print("===== t-SNE ANALYSIS COMPLETE =====")
    wandb.finish()


if __name__ == "__main__":
    main()
