#!/bin/bash

# This works but doesn't release lock until parent pid exits
#DEVICE_ID="/gpu:$(/nfs/ge/bin/request-gpu-device.py)"
DEVICE_ID="/gpu:0"

echo "Requesting device: $DEVICE_ID"

/mnt/nfs/work/momeara/tools/anaconda2/bin/fit_fingerprints \
  --device $DEVICE_ID \
  --summaries_dir /scratch/momeara/train_fingerprint_function-tensorflow_hmdbendo_160630 \
  --output_fp_function_fname /mnt/nfs/work/momeara/sea/DeepSEA/zinc_regression/data/hmdbendo_protomers_160606_net_charge.fp_func \
  --output_training_curve_fname /mnt/nfs/work/momeara/sea/DeepSEA/zinc_regression/data/hmdbendo_protomers_160630_net_charge.fp_func.curve \
  --verbose \
  --train_substances_fname /mnt/nfs/work/momeara/sea/DeepSEA/zinc_regression/data/chembl21_compounds_train_160731_1.tsv \
  --train_substances_field_delim '	' \
  --train_batch_size 3 \
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
  --validate_substances_fname /mnt/nfs/work/momeara/sea/DeepSEA/zinc_regression/data/chembl21_compounds_validate_160731_1.tsv \
  --validate_substances_field_delim '	' \
  --validate_frequency 20 \
  --validate_batch_size 100 \
  --validate_n_batches 20 \
  --validate_queue_capacity 10000 \
  --validate_queue_min_after_dequeue 5000 \
  --validate_queue_num_threads 5 \
  --validate_queue_seed 0 \
  --test_substances_fname /mnt/nfs/work/momeara/sea/DeepSEA/zinc_regression/data/chembl21_compounds_test_160731_1.tsv \
  --test_substances_field_delim '	' \
  --test_batch_size 100 \
  --test_n_batches 28 \
  --test_queue_capacity 10000 \
  --test_queue_min_after_dequeue 5000 \
  --test_queue_num_threads 5 \
  --test_queue_seed 0 \
  --fp_type neural \
  --fp_length 20 \
  --fp_depth 1 \
  --fp_width 100 \
  --h1_size 100 \
  --l2_penalty .00 \
  --l1_penalty 0.0 \
  --fp_normalize \
  --prediction_layer_sizes 20 100 \



#
#  --validate_substances_fname /mnt/nfs/work/momeara/sea/DeepSEA/zinc_regression/data/chembl21_compounds_validate_160731.tsv \
#  --test_substances_fname /mnt/nfs/work/momeara/sea/DeepSEA/zinc_regression/data/chembl21_compounds_test_160731.tsv \
