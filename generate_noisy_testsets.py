# generate_noisy_datasets.py

import torch
from torchvision import datasets, transforms
import argparse
import os
from tqdm import tqdm

from quanv_layer import quanv_layer  # We need this to process the noisy data


# --- Noise Application Functions (copied from previous script) ---
def add_gaussian_noise(image, std):
    noise = torch.randn_like(image) * std
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)


def add_salt_pepper_noise(image, amount):
    """
    Adds salt and pepper noise to a grayscale tensor image of shape (1, H, W).
    Image values are assumed in [0,1].
    """
    if amount == 0.0:
        return image

    s_vs_p = 0.5
    noisy_image = image.clone()
    _, H, W = noisy_image.shape

    # Number of pixels to alter
    num_salt = int(amount * H * W * s_vs_p)
    num_pepper = int(amount * H * W * (1.0 - s_vs_p))

    # Salt noise
    coords_h = torch.randint(0, H, (num_salt,))
    coords_w = torch.randint(0, W, (num_salt,))
    noisy_image[0, coords_h, coords_w] = 1.0

    # Pepper noise
    coords_h = torch.randint(0, H, (num_pepper,))
    coords_w = torch.randint(0, W, (num_pepper,))
    noisy_image[0, coords_h, coords_w] = 0.0

    return noisy_image


def main():
    parser = argparse.ArgumentParser(
        description="Generate noisy classical and quantum-processed test sets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fmnist", "kmnist", "cifar10"],
        required=True,
    )
    parser.add_argument(
        "--noise_type", type=str, choices=["gaussian", "salt_pepper"], required=True
    )

    # NOTE: Default values for the gaussian stds and salt pepper amounts in evaluate_robustness.py
    #
    # parser.add_argument(
    #     "--gaussian_stds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5]
    # )
    # parser.add_argument(
    #     "--salt_pepper_amounts",
    #     nargs="+",
    #     type=float,
    #     default=[0.05, 0.1, 0.15, 0.2, 0.25],
    # )
    parser.add_argument(
        "--noise_level",
        type=float,
        required=True,
        help="Standard deviation for Gaussian, or proportion for Salt & Pepper.",
    )
    args = parser.parse_args()

    # --- 1. Load the original clean test set ---
    print(f"Loading clean {args.dataset.upper()} test set...")
    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == "mnist":
        clean_test_set = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset == "fmnist":
        clean_test_set = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset == "kmnist":  # kmnist
        clean_test_set = datasets.KMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset == "cifar10":
        cifar_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),  # RGB → grayscale
                transforms.Resize(28),  # resize shortest side to 28
                transforms.CenterCrop(28),  # crop to exactly 28x28
                transforms.ToTensor(),
            ]
        )
        clean_test_set = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=cifar_transform
        )
    # --- 2. Generate the classical noisy dataset ---
    print(
        f"Generating CLASSICAL noisy dataset with {args.noise_type} noise at level {args.noise_level}..."
    )
    noisy_classical_images = []
    noisy_labels = []

    for image, label in tqdm(clean_test_set, desc="Applying noise"):
        if args.noise_type == "gaussian":
            noisy_image = add_gaussian_noise(image, args.noise_level)
        else:  # salt_pepper
            noisy_image = add_salt_pepper_noise(image, args.noise_level)
        noisy_classical_images.append(noisy_image)
        noisy_labels.append(label)

    # Convert to a single tensor for easier saving
    noisy_classical_images_tensor = torch.stack(noisy_classical_images)
    noisy_labels_tensor = torch.tensor(noisy_labels)

    # Define output path and save
    output_dir = "./data/noisy_test_sets"
    os.makedirs(output_dir, exist_ok=True)
    classical_path = os.path.join(
        output_dir, f"classical_{args.dataset}_{args.noise_type}_{args.noise_level}.pt"
    )
    torch.save((noisy_classical_images_tensor, noisy_labels_tensor), classical_path)
    print(f"Saved classical noisy test set to {classical_path}")

    # --- 3. Generate the quantum-processed noisy dataset ---
    print("Generating QUANTUM-PROCESSED noisy dataset...")
    processed_noisy_images = []
    for noisy_image in tqdm(noisy_classical_images, desc="Quantum processing"):
        # quanv_layer expects a batch dimension, so we add and remove it
        processed_image = quanv_layer(noisy_image.unsqueeze(0))
        processed_noisy_images.append(processed_image)

    processed_noisy_images_tensor = torch.stack(processed_noisy_images)
    # Labels are the same as before

    quanv_path = os.path.join(
        output_dir, f"quanv_{args.dataset}_{args.noise_type}_{args.noise_level}.pt"
    )
    torch.save((processed_noisy_images_tensor, noisy_labels_tensor), quanv_path)
    print(f"Saved quantum-processed noisy test set to {quanv_path}")
    print("Generation complete.")


if __name__ == "__main__":
    main()
