# run_error_analysis.py

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


def get_predictions(model, loader, device):
    """
    Runs a model on a data loader and returns all true labels and predictions.
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(
            loader, desc=f"Getting predictions from {model.__class__.__name__}"
        ):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def get_class_labels(dataset_name):
    """Returns the class labels for a given dataset for plotting."""
    if dataset_name in ["mnist", "kmnist"]:
        return [str(i) for i in range(10)]
    elif dataset_name == "fmnist":
        return [
            "T-shirt",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    elif dataset_name == "cifar10":
        return (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
    return None


def main():
    parser = argparse.ArgumentParser(description="Run confusion matrix error analysis.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fmnist", "kmnist", "cifar10"],
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for running predictions.",
    )
    args = parser.parse_args()

    wandb.init(
        project="quanvolutional-nn-mnist",
        job_type="error-analysis",
        config=args,
        name=f"confusion_matrix_{args.dataset}",
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Load Pre-trained Models ---
    model_dir = "./saved_models"
    classical_model_path = os.path.join(model_dir, f"{config.dataset}_cnn.pt")
    qnn_model_path = os.path.join(model_dir, f"{config.dataset}_qnn.pt")

    print("Loading pre-trained models...")
    classical_model = ClassicalCNN()
    classical_model.load_state_dict(torch.load(classical_model_path))

    qnn_model = QNN()
    qnn_model.load_state_dict(torch.load(qnn_model_path))

    # --- 2. Load the correct test dataset for each model ---
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
    elif config.dataset == "kmnist":  # kmnist
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

    classical_loader = DataLoader(classical_test_set, batch_size=config.batch_size)
    qnn_loader = DataLoader(qnn_test_set, batch_size=config.batch_size)

    # --- 3. Get Predictions from Both Models ---
    classical_true, classical_pred = get_predictions(
        classical_model, classical_loader, device
    )
    qnn_true, qnn_pred = get_predictions(qnn_model, qnn_loader, device)

    # --- 4. Compute and Plot Confusion Matrices ---
    print("Generating confusion matrix plots...")
    class_labels = get_class_labels(config.dataset)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Plot for Classical CNN
    cm_classical = confusion_matrix(classical_true, classical_pred)
    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=cm_classical, display_labels=class_labels
    )
    disp1.plot(ax=ax1, cmap="Blues", xticks_rotation="vertical")
    ax1.set_title("Classical CNN Confusion Matrix")

    # Plot for Quanv-CNN
    cm_qnn = confusion_matrix(qnn_true, qnn_pred)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_qnn, display_labels=class_labels)
    disp2.plot(ax=ax2, cmap="Oranges", xticks_rotation="vertical")
    ax2.set_title("Quanv-CNN Confusion Matrix")

    fig.suptitle(f"Error Analysis on {config.dataset.upper()} Test Set", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 5. Log to W&B ---
    plt.savefig(f"confusion_matrix_{config.dataset}.png")
    wandb.log({"confusion_matrix_plot": wandb.Image(fig)})
    plt.close(fig)

    print("===== ERROR ANALYSIS COMPLETE =====")
    wandb.finish()


if __name__ == "__main__":
    main()
