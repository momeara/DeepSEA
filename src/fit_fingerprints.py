#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from neuralfingerprint import load_data

from six.moves import xrange  # pylint: disable=redefined-builtin




def fit_fingerprints(
	task_params,
	model_params,
	train_params,
	verbose):

	if verbose:
		print("Building fingerprint function of length {fp_length} as a convolutional network with width {fp_width} and depth {fp_depth} ...".format(**model_params))

	with tf.device(task_params['device']):
		variables = initialize_variables(model_params)

		train_placeholders = initialize_placeholders()
		train_fp = build_fp_network(train_placeholders, variables, model_params)
		train_predictions = build_prediction_network(train_fp, variables, model_params)
		train_loss = build_loss_network(
			train_predictions, train_placeholders, variables, model_params)
		optimizer = build_optimizer(train_loss, train_params)

		eval_placeholders = initialize_placeholders()
		eval_fp = build_fp_network(eval_placeholders, variables, model_params)
		eval_predictions = build_prediction_network(eval_fp, variables, model_params)

		train_summary = build_summary_network(train_loss)

	if verbose:
		print("Loading data from '{data_fname}' with\n\tsmiles column: '{smiles_column}'\n\ttarget column: '{target_column}'\n\tN_train: {N_train}\n\tN_validate: {N_validate}\n\tN_test: {N_test}\n".format(**task_params))

	data = load_data(
		filename=task_params['data_fname'],
		sizes=(task_params['N_train'], task_params['N_validate'], task_params['N_test']),
		input_name=task_params['smiles_column'],
		target_name=task_params['target_column'])

	train_smiles, train_labels = data[0]
	eval_smiles, eval_labels = data[1]
	test_smiles, test_labels = data[2]
	norm_fn, restore_norm_fn = normalize_array(train_labels)

	if verbose:
		print("Begin Tensorflow session ...")
	start_time = time.time()


	training_loss_curve = []
	training_rmse_curve = []
	validation_rmse_curve = []

	session_config = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement=True)

	with tf.Session(config=session_config) as sess:
		sess.run(tf.initialize_all_variables())

		train_writer = tf.train.SummaryWriter(task_params['summaries_dir'] + '/train', sess.graph)
		test_writer = tf.train.SummaryWriter(task_params['summaries_dir'] + '/test')

		n_batches = int(train_params['epochs'] * task_params['N_train']) // train_params['batch_size']
		for step in xrange(n_batches):
			# Compute the offset of the current minibatch in the data.
			# Note that we could use better randomization across epochs.
			offset = (step * train_params['batch_size']) % \
					 (task_params['N_train'] - train_params['batch_size'])
			batch_smiles = train_smiles[offset:(offset + train_params['batch_size']), ...]
			batch_labels = train_labels[offset:(offset + train_params['batch_size'])]
			batch_normed_labels = norm_fn(batch_labels)

			feed_dict = build_feed(batch_smiles, batch_normed_labels, train_placeholders)

			_, loss, predictions, summary = sess.run(
				fetches=[optimizer, train_loss, train_predictions, train_summary],
				feed_dict=feed_dict)

			training_loss_curve += [loss]
			train_rmse = rmse(restore_norm_fn(predictions), batch_labels)
			training_rmse_curve += [train_rmse]

			test_writer.add_summary(summary, step)


			if step % train_params['eval_frequency'] == 0:

#				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#				run_metadata = tf.RunMetadata()
#				summary, _ = sess.run(
#					[train_summary, optimizer],
#					feed_dict=feed_dict(True),
#					options=run_options,
#					run_metadata=run_metadata)
#				train_writer.add_run_metadata(run_metadata, 'step%d' % i)
#				train_writer.add_summary(summary, i)

				elapsed_time = time.time() - start_time
				start_time = time.time()
				print('Minibatch loss: %.3f' % (loss))
				print('Step %d (epoch %.2f), %.1f ms' %
					(step, float(step) * train_params['batch_size'] / task_params['N_train'],
					1000 * elapsed_time / train_params['eval_frequency']))
				print('Minibatch RMSE: %.1f' % train_rmse)

				with tf.device(task_params['device']):
				   validation_predictions = eval_in_batches(
					   sess, eval_smiles, eval_predictions, eval_placeholders, train_params)
				validation_rmse = rmse(restore_norm_fn(validation_predictions), eval_labels)
				validation_rmse_curve += [validation_rmse]
				print('Validation RMSE: %.1f' % validation_rmse)
				print("")
			else:
				validation_rmse_curve += [None]

		test_predictions = eval_in_batches(
			sess, test_smiles, eval_predictions, eval_placeholders, train_params)
		test_RMSE = rmse(restore_norm_fn(test_predictions), test_labels)
		print('Test RMSE: %.1f' % test_RMSE)

		if verbose:
			print("Complete returning ... ")

		return training_loss_curve, training_rmse_curve, validation_rmse_curve
