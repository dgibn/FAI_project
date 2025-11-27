#!/bin/bash

# Setup
DATASET_NAME="CIFAR10"    
SCRIPT_PATH="/users/student/pg/pg23/vaibhav.rathore/D_GCD/DG/project/train.py"
LOG_DIR="/users/student/pg/pg23/vaibhav.rathore/D_GCD/DG/project/logs/bn3/test"
SPLIT="test"
LOG_FILE="$LOG_DIR/$DATASET_NAME.log"
DEVICE_ID=5
# Create log directory if it doesqn't exist
mkdir -p $LOG_DIR

# Run the Python script and log the output
python3 $SCRIPT_PATH --split $SPLIT --dataset_name $DATASET_NAME --device $DEVICE_ID --task_epochs 101 > $LOG_FILE 2>&1 &
echo "Training for $DATASET_NAME begin. Check logs at $LOG_FILE"
