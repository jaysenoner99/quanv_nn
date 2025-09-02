# train_classical.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import (
    datasets,
    transforms,
)  # --- CHANGE: Import torchvision datasets ---
from cnn_model import ClassicalCNN  # --- CHANGE: Import the classical model ---
from tqdm import tqdm
import argparse
import wandb


def prepare_dataset(args):
    dataset = args.dataset
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset == "mnist":
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "fmnist":
        train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "kmnist":
        train_dataset = datasets.KMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.KMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "cifar10":
        cifar_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),  # RGB → grayscale
                transforms.Resize(28),  # resize shortest side to 28
                transforms.CenterCrop(28),  # crop to exactly 28x28
                transforms.ToTensor(),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=cifar_transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=cifar_transform
        )
    return train_dataset, test_dataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # --- 1. Command-Line Argument Parsing (No data path args needed) ---
    parser = argparse.ArgumentParser(
        description="Train a Classical CNN on original MNIST data for baseline comparison."
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training, validation, and testing.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fmnist", "kmnist", "cifar10"],
        default="mnist",
        help="Dataset to preprocess",
    )
    args = parser.parse_args()
    full_train_dataset, test_dataset = prepare_dataset(args)
    # --- 2. Initialize Weights & Biases ---
    wandb.init(
        project="quanvolutional-nn-mnist",
        job_type="classical-baseline",  # Add a job_type for easy filtering in W&B
        config={
            "learning_rate": args.lr,
            "architecture": "Classical-CNN-Baseline",  # --- CHANGE: Note the architecture ---
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
    )
    config = wandb.config

    torch.manual_seed(42)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    print("Creating Dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(
        f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples."
    )

    # --- 4. Model, Optimizer, and Loss Function ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClassicalCNN().to(device)  # --- CHANGE: Instantiate the ClassicalCNN ---

    cnn_params = count_parameters(model)
    print(f"CNN has {cnn_params:,} trainable parameters.")
    wandb.summary["parameters"] = cnn_params
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # --- 5. Training and Validation Loop (Identical Logic) ---
    print("Starting training...")
    for epoch in range(config.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]", leave=False
        )
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loop = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Val]", leave=False
        )
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{config.epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
        )

        # --- 6. Logging to W&B (Identical Logic) ---
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "validation_accuracy": val_accuracy,
            }
        )

    print("\nFinished Training.")
    os.makedirs("./saved_models", exist_ok=True)
    model_path = f"./saved_models/{config.dataset}_cnn.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # --- 7. Final Test Routine (Identical Logic) ---
    print("Running final test on the test set...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_test_loss = test_loss / len(test_loader)
    final_test_accuracy = 100 * correct / total

    print(
        f"\nTest Results -> Loss: {final_test_loss:.4f}, Accuracy: {final_test_accuracy:.2f}%"
    )

    # Log final metrics to W&B summary
    wandb.summary["test_loss"] = final_test_loss
    wandb.summary["test_accuracy"] = final_test_accuracy
    wandb.log(
        {
            "test_loss": final_test_loss,
            "test_acc": final_test_accuracy,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
