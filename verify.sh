#! /bin/bash

train_id="verification"
num_epochs=200

python train_twin_head_segformer.py --seed 109 --id ${train_id} --num_epochs ${num_epochs}
python inference.py --model_path checkpoints/${train_id}_${num_epochs}.pt --output_name ${train_id}_${num_epochs}_submission
