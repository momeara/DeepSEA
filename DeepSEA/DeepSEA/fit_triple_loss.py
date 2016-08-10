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
	initialize_session,
	checkpoint_session,
)

from DeepSEA.queue_substances import (
	smiles_triple_batch_queue,
	smiles_to_flat_substances_network,
)

from DeepSEA.model import (
	initialize_fingerprint_variables,
	load_variables,
	build_fingerprint_summary_network,
	build_neural_fps_network,
	build_morgan_fps_network,
	build_continuous_jaccard_distance_network,
	build_triple_score_network,
	build_triple_loss_network,
	build_loss_network,
	build_optimizer,
)


def eval_in_batches(sess, coord, threads, score, n_batches):
	scores_eval = []
	try:
		for step in xrange(n_batches):
			if coord.should_stop(): break
			s = sess.run(fetches=[score])
			scores_eval.append(s)
	except tf.errors.OutOfRangeError:
		pass


	scores_eval = np.concatenate(scores_eval)
	return scores_eval


def build_score_network(variables, eval_params, model_params):
	smiles, smiles_plus, smiles_minus = smiles_triple_batch_queue(eval_params)

	substances = smiles_to_flat_substances_network(smiles, eval_params)
	substances_plus = smiles_to_flat_substances_network(smiles_plus, eval_params)
	substances_minus = smiles_to_flat_substances_network(smiles_minus, eval_params)

	fps = build_neural_fps_network(substances, variables, model_params)
	fps_plus = build_neural_fps_network(substances_plus, variables, model_params)
	fps_minus = build_neural_fps_network(substances_minus, variables, model_params)

	distance_to_plus = build_continuous_jaccard_distance_network(fps, fps_plus)
	distance_to_minus = build_continuous_jaccard_distance_network(fps, fps_minus)

	triple_score = build_triple_score_network(
		distance_to_plus, distance_to_minus, model_params)

	return fps, triple_score



def fit_triple_loss(
	task_params,
	model_params,
	train_params,
	validate_params,
	test_params):

	if task_params['verbose']:
		print("Building fingerprint function of length {fp_length} as a convolutional network with width {fp_width} and depth {fp_depth} ...".format(**model_params))

	with tf.device(task_params['device']):
		fingerprint_variables = initialize_fingerprint_variables(train_params, model_params)

		train_fps, train_score = build_score_network(
			fingerprint_variables, train_params, model_params)
		train_loss = build_triple_loss_network(train_score, fingerprint_variables, model_params)
		optimizer = build_optimizer(train_loss, train_params)
		build_fingerprint_summary_network(
			train_fps, train_loss, fingerprint_variables, model_params)

		validate_fps, validate_score = build_score_network(
			fingerprint_variables, validate_params, model_params)
		test_fps, test_score = build_score_network(
			fingerprint_variables, test_params, model_params)

	session_config = tf.ConfigProto(
		allow_soft_placement=True,
		log_device_placement=False)

	with tf.Session(config=session_config) as sess:
		coord, threads, saver, train_writer, summaries = initialize_session(sess, task_params)

		validate_mean_score_curve = []

		try:
			start_time = time.time()
			for train_step in xrange(train_params['n_batches']):
				if coord.should_stop(): break

				_, score, loss, summary = sess.run(
					fetches=[optimizer, train_score, train_loss, summaries])

				train_writer.add_summary(summary, train_step)
				checkpoint_session(train_step, saver, sess, task_params)


				if train_step % validate_params['validate_frequency'] == 0:

					elapsed_time = time.time() - start_time
					start_time = time.time()
					print('Minibatch %d: %.1f ms' %
						(train_step, 1000 * elapsed_time / validate_params['validate_frequency']))
					print("Minibatch mean score: {}".format(np.mean(score)))
					print('Minibatch mean loss: {}'.format(np.mean(loss)))

					with tf.device(task_params['device']):
						validate_score_eval = eval_in_batches(
							sess, coord, threads,
							validate_score,
							validate_params['n_batches'])

					validate_mean_score = np.mean(validate_score_eval)
					validate_mean_score_curve += validate_mean_score
					print('Validate Mean Score: {}'.format(validate_mean_score))
					print("")

		except tf.errors.OutOfRangeError:
			pass

		with tf.device(task_params['device']):
			test_score_eval = eval_in_batches(
				sess, coord, threads,
				test_score,
				test_params['n_batches'])
		print('Test Mean Score: %.4f' % test_mean_score)

		coord.join(threads)

		if task_params['verbose']:
			print("Complete returning ... ")



		return validate_mean_score_curve


