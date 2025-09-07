#!/bin/bash

# Script to run start_poses with config file selection
# Usage: ./run_start_poses.sh <config_name> <motion> [case_name]
# Example: ./run_start_poses.sh single_push_rope push
# Example: ./run_start_poses.sh single_lift_sloth lift my_custom_case

# Check if at least 2 arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_name> <motion> [case_name]"
    echo ""
    echo "Available config files:"
    ls PhysTwin/data_generation/config_*.yaml | sed 's/.*config_\(.*\)\.yaml/\1/' | sort
    echo ""
    echo "Motion options: push, lift"
    echo ""
    echo "Examples:"
    echo "  $0 single_push_rope push"
    echo "  $0 single_lift_sloth lift"
    echo "  $0 single_push_rope push my_custom_case"
    exit 1
fi

CONFIG_NAME=$1
MOTION=$2
CASE_NAME=$3

# Construct the config file path
CONFIG_FILE="PhysTwin/data_generation/config_${CONFIG_NAME}.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo ""
    echo "Available config files:"
    ls PhysTwin/data_generation/config_*.yaml | sed 's/.*config_\(.*\)\.yaml/\1/' | sort
    exit 1
fi

# Check if motion is valid
if [ "$MOTION" != "push" ] && [ "$MOTION" != "lift" ]; then
    echo "Error: Motion must be 'push' or 'lift', got '$MOTION'"
    exit 1
fi

echo "Running start_poses with:"
echo "  Config: $CONFIG_FILE"
echo "  Motion: $MOTION"
if [ -n "$CASE_NAME" ]; then
    echo "  Case name: $CASE_NAME"
fi
echo ""

# Run the start_poses script
if [ -n "$CASE_NAME" ]; then
    python PhysTwin/data_generation/start_poses.py --config "$CONFIG_FILE" --motion "$MOTION" --case_name "$CASE_NAME"
else
    python PhysTwin/data_generation/start_poses.py --config "$CONFIG_FILE" --motion "$MOTION"
fi
