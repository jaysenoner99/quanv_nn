# trainer.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


def train_and_evaluate(model, train_loader, test_loader, config, device):
    """
    A reusable function to train, validate, and test a model.
    Returns the final test accuracy.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    # --- Training and Validation Loop ---
    for epoch in range(config.epochs):
        train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}") as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

    # --- Final Test Routine ---
    print("Running final test...")
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

    final_test_accuracy = 100 * correct / total
    print(f"Final Test Accuracy: {final_test_accuracy:.2f}%")
    return final_test_accuracy
