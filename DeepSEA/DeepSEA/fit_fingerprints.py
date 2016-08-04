#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from DeepSEA.util import (
	rmse,
)

from DeepSEA.queue_substances import (
	smiles_labels_batch_queue,
	smiles_to_flat_substances_network,
)

from DeepSEA.model import (
	initialize_variables,
	load_variables,
	build_summary_network,
	build_neural_fps_network,
	build_morgan_fps_network,
	build_normed_prediction_network,
	build_loss_network,
	build_optimizer,
)


def eval_in_batches(sess, coord, threads, predictions, labels, n_batches):
	predictions_eval = []
	labels_eval = []
	try:
		for step in xrange(n_batches):
			if coord.should_stop(): break
			p, l = sess.run(fetches=[predictions, labels])
			predictions_eval.append(p)
			labels_eval.append(l)
	except tf.errors.OutOfRangeError:
		pass


	predictions_eval = np.concatenate(predictions_eval)
	labels_eval = np.concatenate(labels_eval)
	return predictions_eval, labels_eval



def fit_fingerprints(
	task_params,
	model_params,
	train_params,
	validate_params,
	test_params):

	if task_params['verbose']:
		print("Building fingerprint function of length {fp_length} as a convolutional network with width {fp_width} and depth {fp_depth} ...".format(**model_params))

	with tf.device(task_params['device']):
		variables = initialize_variables(train_params, model_params)

		saver = tf.train.Saver()

		train_smiles, train_labels = smiles_labels_batch_queue(train_params)
		if model_params['fp_type'] == 'neural':
			train_substances = smiles_to_flat_substances_network(
				train_smiles, train_params)
			train_fps = build_neural_fps_network(
				train_substances, variables, model_params)
		elif model_params['fp_type'] == 'morgan':
			train_fps = build_morgan_fps_network(
				train_smiles, train_params, model_params)
		else:
			raise Exception("Unrecognized fp_type {}".format(model_params['fp_type']))

		train_normed_predictions = build_normed_prediction_network(
			train_fps, variables, model_params)
		train_predictions, train_loss = build_loss_network(
			train_normed_predictions, train_labels, variables, model_params)
		optimizer = build_optimizer(train_loss, train_params)
		train_summary = build_summary_network(train_fps, train_loss, variables, model_params)

		validate_smiles, validate_labels = smiles_labels_batch_queue(validate_params)
		if model_params['fp_type'] == 'neural':
			validate_substances = smiles_to_flat_substances_network(
				validate_smiles, validate_params)
			validate_fps = build_neural_fps_network(
				validate_substances, variables, model_params)
		elif model_params['fp_type'] == 'morgan':
			validate_fps = build_morgan_fps_network(
				validate_smiles, validate_params, model_params)
		else:
			raise Exception("Unrecognized fp_type {}".format(model_params['fp_type']))

		validate_normed_predictions = build_normed_prediction_network(
			validate_fps, variables, model_params)
		validate_predictions, validate_loss = build_loss_network(
			validate_normed_predictions, validate_labels, variables, model_params)
#		validate_summary = build_summary_network(validate_fps, validate_loss, variables, model_params)

		test_smiles, test_labels = smiles_labels_batch_queue(test_params)
		if model_params['fp_type'] == 'neural':
			test_substances = smiles_to_flat_substances_network(
				test_smiles, test_params)
			test_fps = build_neural_fps_network(
				test_substances, variables, model_params)
		elif model_params['fp_type'] == 'morgan':
			test_fps = build_morgan_fps_network(test_smiles, test_params, model_params)
		else:
			raise Exception("Unrecognized fp_type {}".format(model_params['fp_type']))

		test_normed_predictions = build_normed_prediction_network(
			test_fps, variables, model_params)
		test_predictions, test_loss = build_loss_network(
			test_normed_predictions, test_labels, variables, model_params)


	if task_params['verbose']:
		print("Queuing training data from '{substances_fname}'\n".format(**train_params))
		print("Queuing validation data from '{substances_fname}'\n".format(**validate_params))
		print("Queuing test data from '{substances_fname}'\n".format(**test_params))

	if task_params['verbose']:
		print("Begin Tensorflow session ...")
	start_time = time.time()

	training_loss_curve = []
	training_rmse_curve = []
	validate_rmse_curve = []

	session_config = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement=False)

	with tf.Session(config=session_config) as sess:
		if task_params['restore_from_checkpoint']:
			saver.restore(
				sess=sess,
				save_path=task_params['save_path'])
			if task_params['verbose']:
				print("Restoring variables from '{}'".format(task_params['save_path']))
		else:
			sess.run(tf.initialize_all_variables())
			sess.run(tf.initialize_local_variables())

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		if task_params['verbose']:
			print("Initalized tensorflow session ...")

		train_writer = tf.train.SummaryWriter(
			logdir=task_params['summaries_dir'] + '/train_' + time.strftime("%Y%m%d_%H-%M-%S"),
			graph=sess.graph)

		try:
			for train_step in xrange(train_params['n_batches']):
				if coord.should_stop(): break

				_, loss, predictions, labels, summary = sess.run(
					fetches=[optimizer, train_loss, train_predictions, train_labels, train_summary])

				training_loss_curve += [loss]
				train_rmse = rmse(predictions, labels)
				training_rmse_curve += [train_rmse]
				train_writer.add_summary(summary, train_step)

				if train_step % task_params['checkpoint_frequency'] == 0:
					save_path = saver.save(
						sess=sess,
						save_path=task_params['save_path'],
						global_step=train_step)
					if task_params['verbose']:
						print("Saving variables to '{}'".format(save_path))

				if train_step % validate_params['validate_frequency'] == 0:

					elapsed_time = time.time() - start_time
					start_time = time.time()
					print('Minibatch %d: %.1f ms' %
						(train_step, 1000 * elapsed_time / validate_params['validate_frequency']))
					print('Minibatch loss: %.3f' % (loss))
					print('Minibatch RMSE: %.1f' % train_rmse)

					with tf.device(task_params['device']):
						validate_predictions_eval, validate_labels_eval = eval_in_batches(
							sess, coord, threads,
							validate_predictions, validate_labels,
							validate_params['n_batches'])

					validate_rmse = rmse(validate_predictions_eval, validate_labels_eval)
					validate_rmse_curve += [validate_rmse]
					print('Validate RMSE: %.1f' % validate_rmse)
					print("")
				else:
					validate_rmse_curve += [None]
		except tf.errors.OutOfRangeError:
			pass

		with tf.device(task_params['device']):
			test_predictions_eval, test_labels_eval = eval_in_batches(
				sess, coord, threads,
				test_predictions, test_labels,
				test_params['n_batches'])

		test_rmse = rmse(test_predictions_eval, test_labels_eval)
		print('Test RMSE: %.1f' % test_rmse)

		if task_params['verbose']:
			print("Complete returning ... ")

		return training_loss_curve, training_rmse_curve, validate_rmse_curve
