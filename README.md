
# Quanvolutional Neural Networks for Image Recognition

This repository contains the code for a project implementing and evaluating the Quanvolutional Neural Network (QNN) as proposed by Henderson et al. in "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits". This work was completed for the Quantum Machine Learning exam at the University of Florence.

The project goes beyond a simple implementation by conducting a rigorous, multi-faceted comparison against a carefully designed classical baseline on several image datasets, including MNIST, Fashion-MNIST, KMNIST, and CIFAR-10.

Our analysis reveals a compelling trade-off: the QNN exhibits a notable advantage in data-limited scenarios, a finding supported by t-SNE visualizations showing a highly structured feature space. However, this mapping proves to be exceptionally brittle, demonstrating a catastrophic failure in performance on noisy test data, where the classical baseline is significantly more robust.

**All experimental results are logged and publicly available on [Weights & Biases](httpss://wandb.ai/jaysenoner/quanvolutional-nn-mnist/workspace?nw=nwuserjaysenoner1999).**

## Key Features & Experiments
- **Quanvolutional Layer:** Implementation of a 4-qubit non-trainable quantum feature extractor using PennyLane.
- **Hybrid & Classical Models:** A deep CNN backend for the QNN and an analogous, comparable Classical CNN baseline built in PyTorch.
- **Data Preprocessing Pipeline:** A script to apply the quanvolutional filter to entire datasets as a one-time preprocessing step.
- **Baseline Performance Analysis:** Training and evaluation of all models on four full, clean datasets.
- **Data Efficiency Study:** A comprehensive experiment comparing model performance when trained on subsets of data (1%, 5%, ..., 100%).
- **Noise Robustness Analysis:** A detailed evaluation of model resilience to both Gaussian and Salt & Pepper noise.
- **Feature Space Visualization:** t-SNE and Confusion Matrix analysis to qualitatively assess and compare the features produced by the quantum and classical models.
- **Architectural Ablation Study:** Training a minimal classifier to test the "raw power" of the extracted features.
- **Quantum Kernel Visualization:** Analysis of the unitary matrix representing the quantum transformation.

## Project Structure

The repository is organized into several key scripts, each with a specific purpose:

```
.
├── cnn_model.py                # Defines the Classical CNN baseline architecture.
├── model.py                    # Defines the QNN and Minimal Classifier architectures.
├── quantum_circuits.py         # Defines the core PennyLane QNode for the quanv circuit.
├── quanv_layer.py              # Helper function that applies the quantum circuit to an image.
│
├── preprocess_data.py          # (Step 1) Pre-processes a dataset with the quanv layer.
├── train.py                    # Trains the full QNN model on pre-processed data.
├── train_classical.py          # Trains the full Classical CNN on original data.
│
├── run_data_efficiency.py      # Experiment: Runs the data efficiency study.
├── generate_noisy_testsets.py  # Experiment: Generates noisy test sets.
├── evaluate_robustness.py      # Experiment: Evaluates models on the noisy test sets.
├── run_tsne_visualization.py   # Experiment: Runs t-SNE analysis and plotting.
├── run_error_analysis.py       # Experiment: Generates confusion matrices.
├── visualize_kernel.py         # Analysis: Visualizes the quantum kernel matrix.
│
├── trainer.py                  # Reusable training and evaluation logic.
├── generate_noisy.sh  # Utility script to automate noisy data generation.
└── README.md                   # This file.
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jaysenoner99/quanv_nn.git
    cd quanv_nn
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```
3. Install all the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Log in to Weights & Biases:**
    You will need a free W&B account to log your experiments.
    ```bash
    wandb login
    ```

## Reproducing the Results: A Step-by-Step Workflow

The experiments are designed to be run in a logical sequence.

### Step 1: Quantum Preprocessing (The Slow Step)

This step applies the quanvolutional filter to an entire dataset and saves the feature maps. This is computationally expensive and needs to be done once for each dataset/quantum configuration.

```bash
# Preprocess MNIST with a 1-layer quantum circuit
python preprocess_data.py --dataset mnist

# Preprocess FMNIST with a 2-layer circuit (requires modifying preprocess_data.py)
# python preprocess_data.py --dataset fmnist 
```
This will create files like `./data/processed_train_mnist.pt`.

### Step 2: Train Baseline Models

Train the full QNN and Classical CNN on 100% of the data. These saved models are required for the noise and visualization experiments.

```bash
# Train the Classical CNN on MNIST
python train_classical.py --dataset mnist --epochs 20

# Train the QNN on the pre-processed MNIST data
python train.py --path-train processed_train_mnist.pt --path-test processed_test_mnist.pt --epochs 20
```
This will create model checkpoints in the `./saved_models/` directory, e.g., `mnist_classical.pt` and `mnist_qnn.pt`.

### Step 3: Run the Analysis Experiments

Once you have the pre-processed data and trained models, you can run the main experiments.

#### A. Data Efficiency Study
This experiment trains models on subsets of data and logs the results.

```bash
# Run the full data efficiency experiment for FMNIST
python run_data_efficiency.py --dataset fmnist
```

#### B. Noise Robustness Study
This is a two-part process: first generate the noisy data, then evaluate.

```bash
# 1. Generate all noisy test sets for MNIST (Gaussian and Salt & Pepper)
./generate_noisy.sh mnist gaussian 0.0 0.1 0.2 0.3 0.4 0.5
./generate_noisy.sh mnist salt_pepper 0.0 0.05 0.1 0.15 0.2 0.25

# 2. Evaluate the pre-trained models on the generated noisy data
python evaluate_robustness.py --dataset mnist
```

#### C. Feature Space and Error Analysis
These scripts use the pre-trained models from Step 2.

```bash
# Generate t-SNE plots for KMNIST
python run_tsne_visualization.py --dataset kmnist

# Generate confusion matrices for FMNIST
python run_error_analysis.py --dataset fmnist
```



## Citation
If you use this work, please cite the original paper:
```
@article{henderson2019quanvolutional,
  title={Quanvolutional neural networks: powering image recognition with quantum circuits},
  author={Henderson, Maxwell and Shakya, Samriddhi and Pradhan, Shashindra and Cook, Tristan},
  journal={arXiv preprint arXiv:1904.04767},
  year={2019}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
```
