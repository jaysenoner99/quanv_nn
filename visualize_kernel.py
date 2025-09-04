# visualize_kernel.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import matplotlib.pyplot as plt
import argparse
import wandb


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the unitary matrix of a random quantum circuit."
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        help="Number of random layers in the circuit (e.g., 1 or 2).",
        default=1,
    )
    args = parser.parse_args()

    # --- 1. W&B Initialization ---
    wandb.init(
        project="quanvolutional-nn-mnist",
        job_type="kernel-visualization",
        config=args,
        name=f"visualize_kernel_{args.n_layers}_layers",
    )
    config = wandb.config

    # --- 2. Setup Circuit Parameters (Must match main experiment) ---
    n_qubits = 4

    # CRITICAL: Use the same random seed as your preprocessing to generate the EXACT same circuit.
    # I am assuming the seed was 0 from our previous scripts. If you used another, change it here.
    np.random.seed(42)

    # Generate the same random parameters used for the kernel
    rand_params = np.random.uniform(high=2 * np.pi, size=(config.n_layers, n_qubits))

    device = qml.device("lightning.qubit", wires=n_qubits)

    # --- 3. Define the Circuit for Visualization ---
    # We define a new circuit here. Instead of encoding data and returning
    # expectation values, it just applies the random layers and returns the final state.
    # This allows qml.matrix() to calculate the unitary of the transformation itself.
    @qml.qnode(device)
    def kernel_circuit(params):
        """A circuit that applies the random layers and returns the state."""
        RandomLayers(params, wires=list(range(n_qubits)))
        return qml.state()

    print(
        f"Computing the {2**n_qubits}x{2**n_qubits} unitary matrix for the {config.n_layers}-layer circuit..."
    )

    # --- 4. Compute the Unitary Matrix ---
    # qml.matrix() is a transform that turns a QNode into a function that returns the unitary matrix.
    # We pass the pre-generated random parameters to this function.
    unitary_matrix = qml.matrix(kernel_circuit)(rand_params)

    # --- 5. Plot the Matrix ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(10, 10))

    # We plot the magnitude of the complex-valued matrix elements
    im = ax.matshow(np.abs(unitary_matrix), cmap="viridis")

    # Adding plot details
    fig.colorbar(im, ax=ax)
    ax.set_title(
        f"Quantum Kernel Visualization ({config.n_layers} Random Layer(s))", fontsize=16
    )
    ax.set_xlabel("Input Basis State Index")
    ax.set_ylabel("Output Basis State Index")

    # --- 6. Log to W&B ---
    plt.savefig(f"./images/quantum_kernel_{config.n_layers}_layers.png")
    wandb.log({f"quantum_kernel_{config.n_layers}_layers": wandb.Image(fig)})
    plt.close(fig)

    print("\n===== KERNEL VISUALIZATION COMPLETE =====")
    wandb.finish()


if __name__ == "__main__":
    main()
