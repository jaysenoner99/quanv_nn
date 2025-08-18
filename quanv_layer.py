# quanv_layer.py

import torch
from quantum_circuits import quanv_circuit


def quanv_layer(image):
    """
    Applies the quanvolutional layer to a single image.

    Args:
        image (torch.Tensor): A single image tensor of shape (1, 1, 28, 28).

    Returns:
        torch.Tensor: The output feature maps of shape (14, 14, 4).
    """
    # Create an output tensor to store the feature maps
    out = torch.zeros((14, 14, 4))

    # Iterate over the image with a 2x2 filter and stride 2
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # Extract the 2x2 patch
            patch = torch.flatten(image[0, 0, j : j + 2, k : k + 2])

            # Apply the quantum circuit to the patch
            q_results = quanv_circuit(patch)

            # Assign the 4 quantum measurement results to the output feature map
            for c in range(4):
                out[j // 2, k // 2, c] = q_results[c]

    return out
