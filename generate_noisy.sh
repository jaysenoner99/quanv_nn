#!/bin/bash
#
# Script to automate the generation of noisy test sets.
# It loops through a list of noise levels and calls the Python script for each.
#
# Usage:
# ./generate_all_noisy_data.sh <dataset> <noise_type> <level1> [level2 level3 ...]
#
# Example for Gaussian noise:
# ./generate_all_noisy_data.sh mnist gaussian 0.0 0.1 0.2 0.3 0.4 0.5
#
# Example for Salt & Pepper noise:
# ./generate_all_noisy_data.sh fmnist salt_pepper 0.0 0.05 0.1 0.15 0.2 0.25
#

# --- 1. Check for the correct number of arguments ---
# We need at least 3: dataset, noise_type, and one noise_level.
if [ "$#" -lt 3 ]; then
  echo "Error: Invalid number of arguments."
  echo "Usage: $0 <dataset> <noise_type> <level1> [level2 ...]"
  echo "Example: $0 mnist gaussian 0.1 0.2 0.3"
  exit 1
fi

# --- 2. Assign arguments to variables for clarity ---
DATASET=$1
NOISE_TYPE=$2
# This magic line stores all arguments from the 3rd one onwards into an array
NOISE_LEVELS=("${@:3}")

# --- 3. Main Loop ---
echo "--- Starting noisy dataset generation for DATASET='${DATASET}' with NOISE_TYPE='${NOISE_TYPE}' ---"
echo ""

for level in "${NOISE_LEVELS[@]}"; do
  echo "----------------------------------------"
  echo "Processing noise level: ${level}"
  echo "----------------------------------------"

  # Execute the Python script with the correct arguments.
  # The backslashes allow us to format the command for readability.
  python generate_noisy_testsets.py \
    --dataset "${DATASET}" \
    --noise_type "${NOISE_TYPE}" \
    --noise_level "${level}"

  # Optional: Check if the Python script ran successfully. If not, exit.
  if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Python script failed for noise level ${level}. Aborting."
    exit 1
  fi
done

echo ""
echo "--- All noisy datasets generated successfully for ${DATASET}! ---"
