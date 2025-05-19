#!/bin/bash

# Directory containing all the figure scripts
SCRIPT_DIR="./scripts"

# Check if the script directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
  echo "Error: Directory '$SCRIPT_DIR' not found!"
  exit 1
fi

# List of scripts to run (add or remove scripts as needed)
# list scripts dir

FIGURE_SCRIPTS=$(ls "$SCRIPT_DIR" | grep -E '\.sh$')
FIGURE_SCRIPTS=($FIGURE_SCRIPTS)


# Loop through each script and execute it
for script in "${FIGURE_SCRIPTS[@]}"; do
  echo "Running $script..."
  # Check if the script exists
  if [ ! -f "$SCRIPT_DIR/$script" ]; then
    echo "Error: Script '$script' not found in '$SCRIPT_DIR'. Skipping..."
    continue
  fi
  
  # Run the script
  bash "$SCRIPT_DIR/$script"
  
  # Check for success or failure
  if [ $? -eq 0 ]; then
    echo "$script executed successfully!"
  else
    echo "Error occurred while executing $script!"
    exit 1
  fi
done

echo "All figure scripts have been executed."
