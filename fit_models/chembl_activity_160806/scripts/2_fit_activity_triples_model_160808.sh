#!/bin/bash

# This works but doesn't release lock until parent pid exits
#DEVICE_ID="/gpu:$(/nfs/ge/bin/request-gpu-device.py)"
DEVICE_ID="/gpu:0"

echo "Requesting device: $DEVICE_ID"

/mnt/nfs/work/momeara/tools/anaconda2/bin/fit_activity_triples \
  --device $DEVICE_ID \
  --summaries_dir /mnt/nfs/work/momeara/sea/DeepSEA/fit_models/chembl_activity_160806/logs \
  --output_training_curve_fname /mnt/nfs/work/momeara/sea/DeepSEA/fit_models/chembl_activity_160806/logs/training_curve.csv \
  --save_path /mnt/nfs/work/momeara/sea/DeepSEA/fit_models/chembl_activity_160806/checkpoints/checkpoint \
  --checkpoint_frequency 1000 \
  --verbose \
  --train_substances_fname /mnt/nfs/work/momeara/sea/DeepSEA/fit_models/chembl_activity_160806/data/activity_triples_160808_grep_clean.tsv \
  --train_substances_field_delim '	' \
  --train_batch_size 100 \
  --train_n_batches 10000 \
  --train_queue_capacity 2000 \
  --train_queue_min_after_dequeue 0 \
  --train_queue_num_threads 5 \
  --train_queue_seed 0 \
  --log_init_scale -4 \
  --log_learning_rate -6 \
  --log_b1 -3 \
  --log_b2 -2 \
  --validate_substances_fname /mnt/nfs/work/momeara/sea/DeepSEA/fit_models/chembl_activity_160806/data/activity_triples_160808_grep_clean.tsv \
  --validate_substances_field_delim '	' \
  --validate_frequency 500 \
  --validate_batch_size 100 \
  --validate_n_batches 10 \
  --validate_queue_capacity 2000 \
  --validate_queue_min_after_dequeue 10 \
  --validate_queue_num_threads 5 \
  --validate_queue_seed 0 \
  --test_substances_fname /mnt/nfs/work/momeara/sea/DeepSEA/fit_models/chembl_activity_160806/data/activity_triples_160808_grep_clean.tsv \
  --test_substances_field_delim '	' \
  --test_batch_size 100 \
  --test_n_batches 100 \
  --test_queue_capacity 2000 \
  --test_queue_min_after_dequeue 10 \
  --test_queue_num_threads 5 \
  --test_queue_seed 0 \
  --fp_type morgan \
  --fp_length 512 \
  --fp_depth 4 \
  --fp_width 20 \
  --h1_size 100 \
  --l2_penalty .01 \
  --l1_penalty 0.0 \
  --score_gap .1
