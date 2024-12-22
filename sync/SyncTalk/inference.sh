#!/bin/bash

# Ensure the script exits on any error
set -e

# Define variables for the arguments
DATA_PATH="data/May"
WORKSPACE="model/trial_May"
ASR_MODEL="ave"
AUDIO_PATH="data/May/aud.wav"

# Run the Python script with the specified arguments
python main.py "$DATA_PATH" \
  --workspace "$WORKSPACE" \
  -O \
  --test \
  --test_train \
  --asr_model "$ASR_MODEL" \
  --portrait \
  --aud "$AUDIO_PATH"
