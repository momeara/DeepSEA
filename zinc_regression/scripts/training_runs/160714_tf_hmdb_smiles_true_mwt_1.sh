#!/bin/bash

# This works but doesn't release lock until parent pid exits
#DEVICE_ID="/gpu:$(/nfs/ge/bin/request-gpu-device.py)"
DEVICE_ID="/gpu:0"

echo "Requesting device: $DEVICE_ID"

/mnt/nfs/work/momeara/tools/anaconda2/bin/python scripts/train_fingerprint_function-tensorflow.py \
  --input_data_fname data/hmdbendo_protomers_160606.csv \
  --output_fp_function_fname data/hmdbendo_protomers_160606_net_charge.fp_func \
  --summaries_dir /scratch/momeara/train_fingerprint_function-tensorflow_hmdbendo_160630 \
  --output_training_curve_fname data/hmdbendo_protomers_160630_net_charge.fp_func.curve \
  --verbose \
  --smiles_column smiles \
  --target_column true_mwt \
  --N_train 6000 \
  --N_validate 1000 \
  --N_test 1000 \
  --device $DEVICE_ID \
  --fp_length 512 \
  --fp_depth 4 \
  --fp_width 20 \
  --h1_size 100 \
  --l2_penalty .01 \
  --l1_penalty 0.0 \
  --fp_normalize \
  --prediction_layer_sizes 512 100 \
  --epochs 15 \
  --batch_size 10 \
  --eval_frequency 100 \
  --log_init_scale -4 \
  --log_learning_rate -4 \
  --log_stepsize -6 \
  --log_b1 -3 \
  --log_b2 -2
