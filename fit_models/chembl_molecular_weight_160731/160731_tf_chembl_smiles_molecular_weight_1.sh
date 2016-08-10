#!/bin/bash

# This works but doesn't release lock until parent pid exits
#DEVICE_ID="/gpu:$(/nfs/ge/bin/request-gpu-device.py)"
DEVICE_ID="/gpu:0"

echo "Requesting device: $DEVICE_ID"

rm -rf logs
mkdir -p logs/checkpoints

/mnt/nfs/work/momeara/tools/anaconda2/bin/fit_fingerprints \
  --device $DEVICE_ID \
  --summaries_dir logs \
  --verbose \
  --train_substances_fname chembl21_compounds_train_clean_160731.tsv \
  --train_substances_field_delim '	' \
  --train_batch_size 100 \
  --train_n_batches 60000 \
  --train_queue_capacity 5000 \
  --train_queue_min_after_dequeue 2000 \
  --train_queue_num_threads 5 \
  --train_queue_seed 0 \
  --log_init_scale -4 \
  --log_learning_rate -6 \
  --log_stepsize -6 \
  --log_b1 -3 \
  --log_b2 -2 \
  --validate_substances_fname chembl21_compounds_validate_clean_160731.tsv \
  --validate_substances_field_delim '	' \
  --validate_frequency 200 \
  --validate_batch_size 100 \
  --validate_n_batches 20 \
  --validate_queue_capacity 10000 \
  --validate_queue_min_after_dequeue 5000 \
  --validate_queue_num_threads 5 \
  --validate_queue_seed 0 \
  --test_substances_fname chembl21_compounds_test_clean_160731.tsv \
  --test_substances_field_delim '	' \
  --test_batch_size 100 \
  --test_n_batches 28 \
  --test_queue_capacity 10000 \
  --test_queue_min_after_dequeue 5000 \
  --test_queue_num_threads 5 \
  --test_queue_seed 0 \
  --fp_type neural \
  --fp_length 512 \
  --fp_depth 4 \
  --fp_width 100 \
  --h1_size 100 \
  --l2_penalty .01 \
  --l1_penalty 0.0 \
  --prediction_layer_sizes 512 100 
