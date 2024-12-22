#!/bin/bash
source activate diffdub
python demo.py --stage1_checkpoint_path 'assets/checkpoints/stage1_state_dict.ckpt' \
    --stage2_checkpoint_path 'assets/checkpoints/stage2_state_dict.ckpt' "$@"