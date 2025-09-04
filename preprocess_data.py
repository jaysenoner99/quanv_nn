import torch
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from quanv_layer import quanv_layer
from tqdm import tqdm
import os


def preprocess_and_save(dataset, file_name):
    """
    Applies the quanv_layer to each image in the dataset and saves the result.
    """
    # Use a DataLoader with batch_size=1 to iterate through the dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    processed_data = []
    print(f"Processing {file_name}...")

    # Use tqdm for a nice progress bar
    for image, label in tqdm(loader):
        # Apply the quantum transformation
        # image shape is (1, 1, 28, 28), which quanv_layer expects
        quanv_output = quanv_layer(image)

        # Append the processed tensor and its label to our list
        processed_data.append((quanv_output, label.item()))

    # Save the entire list of processed data to a file
    torch.save(processed_data, file_name)
    print(f"Saved processed data to {file_name}")


def main():
    # Ensure the data directory exists
    if not os.path.exists("./data"):
        os.makedirs("./data")

    parser = argparse.ArgumentParser(
        description="Preprocess a Dataset using a random quantum circuit"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fmnist", "kmnist", "cifar10","cifar10_2l", "fmnist_2l", "mnist_2l"],
        default="mnist",
        help="Dataset to preprocess",
    )
    args = parser.parse_args()

    # Standard MNIST transformations
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = args.dataset
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
    elif dataset == "cifar10" or dataset == "cifar10_2l":
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
    filename_train = "./data/processed_train_" + dataset + ".pt"
    filename_test = "./data/processed_test_" + dataset + ".pt"

    # Process and save both training and testing data
    preprocess_and_save(train_dataset, filename_train)
    preprocess_and_save(test_dataset, filename_test)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
