# quantum_circuits.py

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import torch

# Define the number of qubits and layers
n_qubits = 4
n_layers = 2  # As in the tutorial

# Use the high-performance lightning simulator
dev = qml.device("lightning.qubit", wires=n_qubits)

# --- FIXED: Generate random parameters ONCE, outside the qnode ---
# This ensures the same circuit is used for every patch (a constant kernel).
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, n_qubits))


@qml.qnode(dev, interface="torch")
def quanv_circuit(phi):
    """The quantum circuit that will be used as a filter.

    Args:
        phi (torch.Tensor): A tensor of 4 pixel values.
    """
    # Encoding of 4 classical input values
    for j in range(n_qubits):
        qml.RY(np.pi * phi[j], wires=j)

    # --- FIX: Use RandomLayers template with the pre-generated parameters ---
    RandomLayers(rand_params, wires=list(range(n_qubits)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
