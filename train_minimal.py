# train_minimal.py

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import argparse
import wandb
import os
from tqdm import tqdm

# Import your models and the reusable trainer
from model import MinimalClassifier
from cnn_model import ClassicalCNN
from trainer import train_and_evaluate


# Custom Dataset for pre-processed Quantum Data
class QuantumProcessedDataset(Dataset):
    def __init__(self, filepath):
        self.data = torch.load(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def generate_classical_features(dataset_name, model_path, device, train: bool):
    """
    Loads a dataset (train or test), passes it through a pre-trained classical
    feature extractor, and returns a TensorDataset of the resulting features.
    """
    mode = "train" if train else "test"
    print(f"Generating classical features for the {mode} set...")

    # Load the pre-trained classical model
    model = ClassicalCNN()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Load the original dataset based on the arguments
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "mnist":
        full_dataset = datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )
    elif dataset_name == "fmnist":
        full_dataset = datasets.FashionMNIST(
            root="./data", train=train, download=True, transform=transform
        )
    elif dataset_name == "kmnist":
        full_dataset = datasets.KMNIST(
            root="./data", train=train, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        cifar_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),  # RGB → grayscale
                transforms.Resize(28),  # resize shortest side to 28
                transforms.CenterCrop(28),  # crop to exactly 28x28
                transforms.ToTensor(),
            ]
        )

        full_dataset = datasets.CIFAR10(
            root="./data", train=train, download=True, transform=cifar_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(full_dataset, batch_size=256)

    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(
            loader, desc=f"Extracting classical features ({mode})"
        ):
            images = images.to(device)
            # We only use the feature_extractor part of the CNN
            features = model.feature_extractor(images)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    features_tensor = torch.cat(all_features)
    labels_tensor = torch.cat(all_labels)

    return TensorDataset(features_tensor, labels_tensor)


def main():
    parser = argparse.ArgumentParser(
        description="Train a minimal classifier on extracted features."
    )
    parser.add_argument(
        "--feature_type", type=str, choices=["quantum", "classical"], required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fmnist", "kmnist", "cifar10"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    wandb.init(
        project="quanvolutional-nn-mnist",
        job_type="ablation-study",
        config=args,
        name=f"minimal_{args.feature_type}_{args.dataset}",
    )
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # --- Data Loading ---
    if config.feature_type == "quantum":
        print("Loading pre-processed quantum features...")
        q_train_path = f"./data/processed_train_{config.dataset}.pt"
        q_test_path = f"./data/processed_test_{config.dataset}.pt"
        full_train_dataset = QuantumProcessedDataset(q_train_path)
        test_dataset = QuantumProcessedDataset(q_test_path)
        input_features = 4 * 14 * 14

    else:  # classical
        model_path = f"./saved_models/{config.dataset}_cnn.pt"
        # --- FIXED: Use the corrected helper function for both train and test ---
        full_train_dataset = generate_classical_features(
            config.dataset, model_path, device, train=True
        )
        test_dataset = generate_classical_features(
            config.dataset, model_path, device, train=False
        )
        input_features = 4 * 14 * 14

    train_loader = DataLoader(
        full_train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # --- Model and Training ---
    print(f"Training Minimal Classifier on {config.feature_type} features...")
    model = MinimalClassifier(input_features=input_features, num_classes=10)

    test_accuracy = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        # Note: We are not passing a val_loader, so our trainer will skip it
    )

    wandb.summary["final_test_accuracy"] = test_accuracy
    print("\n===== MINIMAL MODEL TRAINING COMPLETE =====\n")
    wandb.finish()


if __name__ == "__main__":
    main()
