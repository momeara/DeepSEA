#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

def initialize_session(sess, task_params):
	if task_params['verbose']:
		print("Initalizing tensorflow session ...")

	saver = tf.train.Saver()
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

	logdir=task_params['summaries_dir'] + '/train_' + time.strftime("%Y%m%d_%H-%M-%S")
	train_writer = tf.train.SummaryWriter(logdir=logdir, graph=sess.graph)

	summaries = tf.merge_all_summaries()


	return coord, threads, saver, train_writer, summaries



def checkpoint_session(train_step, saver, sess, task_params):
	if train_step % task_params['checkpoint_frequency'] == 0:
		save_path = saver.save(
			sess=sess,
			save_path=task_params['save_path'],
			global_step=train_step)
		if task_params['verbose']:
			print("Saving variables to '{}'".format(save_path))


def rmse(predictions, labels):
	return np.sqrt(np.mean((labels - predictions)**2))


