# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from model import QNN
from tqdm import tqdm
import argparse  # For command-line arguments
import wandb  # For logging


# --- Custom Dataset (remains the same) ---
class QuantumProcessedDataset(Dataset):
    def __init__(self, filepath):
        self.data = torch.load(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Main Execution ---
def main():
    # --- 1. Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Train a QNN on pre-processed MNIST data."
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
    args = parser.parse_args()

    # --- 2. Initialize Weights & Biases ---
    wandb.init(
        project="quanvolutional-nn-mnist",
        config={
            "learning_rate": args.lr,
            "architecture": "Quanv-CNN",
            "dataset": "MNIST",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
    )
    # For easy access to hyperparameters
    config = wandb.config

    # --- 3. Data Loading and Splitting ---
    torch.manual_seed(42)  # Ensure reproducibility for the split

    # Load the full pre-processed training data
    full_train_dataset = QuantumProcessedDataset("./data/processed_train.pt")

    # Split the training data into training and validation sets (e.g., 80/20 split)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    # Load the pre-processed test data
    test_dataset = QuantumProcessedDataset("./data/processed_test.pt")

    # Create DataLoaders for all three sets
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print(
        f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples."
    )

    # --- 4. Model, Optimizer, and Loss Function ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # --- 5. Training and Validation Loop ---
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

        # --- 6. Logging to W&B ---
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "validation_accuracy": val_accuracy,
            }
        )

    print("\nFinished Training.")

    # --- 7. Final Test Routine ---
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

    wandb.finish()


if __name__ == "__main__":
    main()
