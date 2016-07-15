#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

# adapt neural-fingeprint/examples/regression.py
# to fit fingerprints based on a single endpoint regression

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import sys
import time
import subprocess
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from neuralfingerprint import load_data
from neuralfingerprint.build_convnet import array_rep_from_smiles
from neuralfingerprint.features import num_atom_features, num_bond_features
from neuralfingerprint.mol_graph import degrees
from neuralfingerprint.build_convnet import array_rep_from_smiles


def normalize_array(A):
	mean, std = np.mean(A), np.std(A)
	def norm_fn(X): return (X - mean) / std
	def restore_fn(X): return X * std + mean
	return norm_fn, restore_fn

def rmse(predictions, labels):
	return np.sqrt(np.mean((labels - predictions)**2))


def initialize_variables(model_params):
	variables = {}
	with tf.name_scope("regularization") as scope:
		variables['l2_loss'] = tf.constant(0.0, name="l2_loss")
		variables['l1_loss'] = tf.constant(0.0, name="l1_loss")


	def add_weights(weight_key, shape, op=tf.random_normal):
		weights = tf.Variable(op(shape), name=weight_key)
		variables[weight_key] = weights
		with tf.name_scope("regularization/") as regularization_scope:
			variables['l2_loss'] += tf.nn.l2_loss(weights)
			variables['l1_loss'] += tf.reduce_sum(tf.abs(weights))

	with tf.name_scope("fingerprint") as scope:
		# neural fingerprint layer output weights and biases
		all_layer_sizes = [num_atom_features()] + [model_params['fp_width']] * model_params['fp_depth']
		for layer in range(len(all_layer_sizes)):
			with tf.name_scope("layer_{}".format(layer)):
				add_weights(
					'layer_output_weights_{}'.format(layer),
					[all_layer_sizes[layer], model_params['fp_length']])

				add_weights(
					'layer_output_bias_{}'.format(layer),
					[model_params['fp_length']])

		 # neural fingerprint graph integration layer weights and biases
		in_and_out_sizes = zip(all_layer_sizes[:-1], all_layer_sizes[1:])
		for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
			with tf.name_scope("layer_{}/".format(layer)) as layer_scope:
				 add_weights(
					 "layer_{}_biases".format(layer),
					 [N_cur])

				 add_weights(
					 "layer_{}_self_filter".format(layer),
					 [N_prev, N_cur])

				 add_weights(
					 "layer_{}_neighbor_filter".format(layer),
					 [N_prev + num_bond_features(), N_cur])

				for degree in degrees:
					 add_weights(
						 'layer_{}_neighbor_{}_filter'.format(layer, degree),
						[N_prev + num_bond_features(), N_cur])

	with tf.name_scope("prediction") as scope:
		# prediction network weights and biases
		layer_sizes = model_params['prediction_layer_sizes'] + [1]
		for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			add_weights(
				"prediction_weights_{}".format(i),
				shape)

			add_weights(
				"prediction_biases_{}".format(i),
				[shape[1]])

	return variables

def initialize_placeholders():
	placeholders = {}
	with tf.name_scope("molecules") as scope:
		with tf.name_scope("features") as features_scope:
			 placeholders['atom_features'] = tf.placeholder(
				dtype=tf.float32,
				shape=[None, num_atom_features()],
				name="atom_features")
			placeholders['bond_features'] = tf.placeholder(
				dtype=tf.float32,
				shape=[None, num_bond_features()],
				name="bond_features")
			placeholders['atom_list'] = tf.sparse_placeholder(
				dtype=tf.float32,
				name="atom_list")

		with tf.name_scope("topology") as topology_scope:
			for degree in degrees:
				data_key = 'atom_neighbors_{}'.format(degree)
				placeholders[data_key] = tf.placeholder(
					dtype=tf.int32,
					shape=[None, degree],
					name=data_key)
				data_key = 'bond_neighbors_{}'.format(degree)
				placeholders[data_key] = tf.placeholder(
					dtype=tf.int32,
					shape=[None, degree],
					name=data_key)

	placeholders['labels'] = tf.placeholder(
		dtype=tf.float32,
		shape=None)

	return placeholders


def build_summary_network(loss):
	loss_summary = tf.scalar_summary("loss", loss)
	summaries = tf.merge_all_summaries()
	return summaries

def build_fp_network(placeholders, variables, model_params):

	def matmult_neighbors(atom_features, layer, data, variables, model_params):
		"""
		N_atoms = 14700
		  # with degree 0 = 0
		  # with degree 1 = 4310
		  # with degree 2 = 5283
		  # with degree 3 = 4290
		  # with degree 4 = 817
		  # with degree 5 = 0

		num_atom_features = 20
		num_bond_features = 6

		atom_features: [N_atoms, num_atom_features()]:float32

		atom_neighbor_list: [N_atoms with degree, degree]:int32
		bond_neighbor_list: [N_atoms with degree, degree]:int32

		stacked_neigbors: [N_atoms with degree, degree, num_atom_features + num_bond_features]
		summed_neighbors: [N_atoms with degree, num_atom_features + num_bond_features]
		activations: [N_atoms with degree, num_atom_features]
		return [N_atoms, num_atom_features]

		"""
		with tf.name_scope("neighbor_activations") as scope:
			activations_by_degree = []
			for degree in degrees:
				atom_neighbor_list = placeholders['atom_neighbors_{}'.format(degree)]
				bond_neighbor_list = placeholders['bond_neighbors_{}'.format(degree)]
				neighbor_filter = variables['layer_{}_neighbor_{}_filter'.format(layer, degree)]
				neighbor_features = [
					tf.gather(params=atom_features, indices=atom_neighbor_list),
					tf.gather(params=placeholders['bond_features'], indices=bond_neighbor_list)]
				stacked_neighbors = tf.concat(concat_dim=2, values=neighbor_features)
				summed_neighbors = tf.reduce_sum(stacked_neighbors, reduction_indices=1)
				activations = tf.matmul(summed_neighbors, neighbor_filter)
				activations_by_degree.append(activations)
				activations = tf.concat(concat_dim=0, values=activations_by_degree, name="activations")
			return activations

	def update_layer(atom_features, layer, placeholders, variables):
		with tf.name_scope("layer_{}/".format(layer)) as update_layer_scope:
			layer_bias		= variables["layer_{}_biases".format(layer)]
			layer_self_filter = variables["layer_{}_self_filter".format(layer)]
			self_activations = tf.matmul(atom_features, layer_self_filter)
			neighbor_activations = matmult_neighbors(
				atom_features, layer, placeholders, variables, model_params)
			activations = tf.nn.bias_add(tf.add(neighbor_activations, self_activations), layer_bias)
			activations_mean, activations_variance = tf.nn.moments(activations, [0], keep_dims=True)
			activations = tf.nn.batch_normalization(
				activations, activations_mean, activations_variance,
				offset=None, scale=None, variance_epsilon=1e-3)
			activations = tf.nn.relu(activations, name="activations")
			return activations

	def write_to_fingerprint(atom_features, layer, placeholders, variables):
		"""
		N_atoms = 14700 (for example)
		N_compounds = 800 (for example)
		num_atom_features = 20

		atom_features: [N_atoms, num_atom_features]
		atom_list: Sparse[N_compounds, N_atoms]

		hidden: [N_atoms, fp_length]
		atom_outputs: [N_atoms, fp_length]
		layer_outputs: [N_compounds, fp_length]
		"""
		with tf.name_scope("layer_{}/".format(layer)) as scope:
			out_weights = variables['layer_output_weights_{}'.format(layer)]
			out_bias	= variables['layer_output_bias_{}'.format(layer)]
			hidden = tf.nn.bias_add(tf.matmul(atom_features, out_weights), out_bias)
			atom_outputs = tf.nn.softmax(hidden)
			layer_output = tf.sparse_tensor_dense_matmul(
				placeholders['atom_list'], atom_outputs, name=scope)
			return layer_output

	with tf.name_scope("fingerprint/") as fingerprint_scope:
		atom_features = placeholders['atom_features']
		fps = write_to_fingerprint(atom_features, 0, placeholders, variables)

		num_hidden_features = [model_params['fp_width']] * model_params['fp_depth']
		for layer in xrange(len(num_hidden_features)):
			atom_features = update_layer(atom_features, layer, placeholders, variables)
			fps += write_to_fingerprint(atom_features, layer+1, placeholders, variables)

		return fps

def build_prediction_network(fps, variables, model_params):
	with tf.name_scope("prediction/") as scope:
		hidden = fps
		layer_sizes = model_params['prediction_layer_sizes'] + [1]
		for layer in range(len(layer_sizes) - 1):
			weights = variables['prediction_weights_{}'.format(layer)]
			biases = variables['prediction_biases_{}'.format(layer)]
			activations = tf.nn.bias_add(tf.matmul(hidden, weights), biases, name="activations")
			if layer < len(layer_sizes) - 2:
				activations_mean, activations_variance = tf.nn.moments(
					activations, [0], keep_dims=True)
				activations = tf.nn.batch_normalization(
					activations, activations_mean, activations_variance,
					offset=None, scale=None, variance_epsilon=1e-3)
			hidden = tf.nn.relu(activations)
		return tf.squeeze(hidden, name="predictions")


def build_loss_network(
	predictions,
	placeholders,
	variables,
	model_params):

	with tf.name_scope("loss") as loss_scope:
		# http://stackoverflow.com/questions/33846069/how-to-set-rmse-cost-function-in-tensorflow
		return tf.sqrt(tf.reduce_mean((predictions - placeholders['labels'])**2)) \
			+ model_params['l2_penalty'] * variables['l2_loss'] \
			+ model_params['l1_penalty'] * variables['l1_loss']


def build_optimizer(loss, train_params):
	with tf.name_scope("optimizer") as optimizer_scope:
		batch = tf.Variable(0.0)
		learning_rate = tf.constant(np.exp(train_params['log_learning_rate']))
		beta1 = tf.constant(np.exp(train_params['log_b1']))
		beta2 = tf.constant(np.exp(train_params['log_b2']))
		adam = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
		optimizer = adam.minimize(loss, global_step=batch)
		return optimizer


def build_feed(smiles, labels, placeholders):
	data = array_rep_from_smiles(tuple(smiles))

	feed_dict = {}
	feed_dict[placeholders['atom_features']] = data['atom_features']
	feed_dict[placeholders['bond_features']] = data['bond_features']

	indices = []
	for compound, atoms in enumerate(data['atom_list']):
		indices += [ [compound, atom] for atom in atoms]
	indices = np.array(indices, dtype=np.int64)
	values = np.ones(indices.shape[0], dtype=np.float32)
	shape = np.array([len(data['atom_list']), len(data['atom_features'])])
	feed_dict[placeholders['atom_list']] = tf.SparseTensorValue(indices, values, shape)

	for degree in degrees:
		atom_neighbors = data[('atom_neighbors', degree)]
		if len(atom_neighbors) == 0: atom_neighbors.shape = [0,degree]
		feed_dict[placeholders['atom_neighbors_{}'.format(degree)]] = atom_neighbors

		bond_neighbors = data[('bond_neighbors', degree)]
		if len(bond_neighbors) == 0: bond_neighbors.shape = [0,degree]
		feed_dict[placeholders['bond_neighbors_{}'.format(degree)]] = bond_neighbors

	if labels is not None:
		feed_dict[placeholders['labels']] = labels
	return feed_dict


# adapted from tensorflow/models/image/mnist/convolutional.py
# Small utility function to evaluate a dataset by feeding batches of data to
# {eval_data} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
def eval_in_batches(sess, smiles, eval_predictions, eval_placeholders, train_params):
	"""Get all predictions for a dataset by running it in small batches."""
	size = smiles.shape[0]
	if size < train_params['eval_batch_size']:
		raise ValueError("batch size for evals larger than dataset: %d" % size)
	predictions = np.ndarray(shape=[size], dtype=np.float32)
	for begin in xrange(0, size, train_params['eval_batch_size']):
		end = begin + train_params['eval_batch_size']
		if end <= size:
			feed_dict = build_feed(smiles[begin:end], None, eval_placeholders)
			predictions[begin:end] = sess.run(eval_predictions, feed_dict=feed_dict)
		else:
			feed_dict = build_feed(
				smiles[-train_params['eval_batch_size']:], None, eval_placeholders)
			batch_predictions = sess.run(eval_predictions, feed_dict=feed_dict)
			predictions[begin:] = batch_predictions[begin - size:]
	return predictions



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
		train_fps = build_fp_network(train_placeholders, variables, model_params)
		train_predictions = build_prediction_network(train_fps, variables, model_params)
		train_loss = build_loss_network(
			train_predictions, train_placeholders, variables, model_params)
		optimizer = build_optimizer(train_loss, train_params)

		eval_placeholders = initialize_placeholders()
		eval_fps = build_fp_network(eval_placeholders, variables, model_params)
		eval_predictions = build_prediction_network(eval_fps, variables, model_params)

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


def main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr):
	parser = ArgumentParser("Train a neural fingerprint function")

	# paths etc.
	parser.add_argument("--input_data_fname", help="Comma separated value file of substance activity data. After a header row, each row represents a substance and having columns identified by --smiles_column and --activity_column", default="data.csv")
#	parser.add_argument("--output_fp_function_fname", help="Name of fingerprint function output file", default="fp_function.pickle")
	parser.add_argument("--summaries_dir", help="Name of directory where summary data should be deposited", default="logs")
	parser.add_argument("--output_training_curve_fname", help="Name of training curve output file", default="training_curve.tsv")
	parser.add_argument("--verbose", default=False, action='store_true', help="Report verbose output")

	#task_params
	parser.add_argument("--smiles_column", help="Name of substance smiles column.", default="smiles")
	parser.add_argument("--target_column", help="Name of substance target column.", default="target")
	parser.add_argument("--N_train", help="Number of substance to use for model training.", default=80, type=int)
	parser.add_argument("--N_validate", help="Number of substances to use for model validation.", default=20, type=int)
	parser.add_argument("--N_test", help="Number of substances to use for model testing.", default=20, type=int)
	parser.add_argument("--device", help="Specify the device to use.", default='cpu:0')
#	parser.add_argument("--seed", help="Random seed used in training.", default=0, type=int)


	# model params
	parser.add_argument("--fp_length", help="Number of elements in the fingerprint vector", default=512, type=int)
	parser.add_argument("--fp_depth", help="Depth of fingerprint neural network", default=4, type=int)
	parser.add_argument("--fp_width", help="Width of fingerprint neural network", default=20, type=int)
	parser.add_argument("--h1_size", help="Size of hidden layer of network on top of fingerprints", default=100, type=int)
	parser.add_argument("--l2_penalty", help="scale of l2 regularization factor for loss in neural network", default=.01, type=float)
	parser.add_argument("--l1_penalty", help="scale of l1 regularization factor for loss in neural network", default=0.0, type=float)
	parser.add_argument("--prediction_layer_sizes", help="vector of layer sizes for fingerprint neural network", default=[512, 100], type=int, nargs='+')

	# train params
	parser.add_argument("--epochs", help="Number of training epochs", default = 10, type=int)
	parser.add_argument("--batch_size", help="Training data batch size", default=100, type=int)
	parser.add_argument("--eval_batch_size", help="Batch size when evluating performance", default=64, type=int)
	parser.add_argument("--eval_frequency", help="How often to evaluate performance", default=10, type=int)
#	parser.add_argument("--log_init_scale", help="Training log initial scale", default=-4, type=float)
	parser.add_argument("--log_learning_rate", help="Training log learning rate", default=-6, type=float)
	parser.add_argument("--log_b1", help="Training log Adam optimizer parameter b1", default=-3, type=float)
	parser.add_argument("--log_b2", help="Training log Adam optimizer parameter b2", default=-2, type=float)

	params, others = parser.parse_known_args(args)

	task_params = dict(
		data_fname = params.input_data_fname,
		summaries_dir = params.summaries_dir,
		N_train = params.N_train,
		N_validate = params.N_validate,
		N_test = params.N_test,
		smiles_column = params.smiles_column,
		target_column = params.target_column,
		device = params.device)
#		seed = params.seed)

	model_params = dict(
		fp_length = params.fp_length,
		fp_depth = params.fp_depth,
		fp_width = params.fp_width,
		h1_size = params.h1_size,
		l2_penalty = params.l2_penalty,
		l1_penalty = params.l1_penalty,
		prediction_layer_sizes = params.prediction_layer_sizes)
#		nll_func_name = params.nll_func_name,
#		nll_func = load_nll_func(params.nll_func_name),


	print("GIT REVISION: {}".format(subprocess.check_output(['git', 'rev-parse', 'HEAD'])))

	train_params = dict(
		epochs = params.epochs,
		batch_size = params.batch_size,
		eval_batch_size = params.eval_batch_size,
		eval_frequency = params.eval_frequency,
#		log_init_scale = params.log_init_scale,
		log_learning_rate = params.log_learning_rate,
		log_b1 = params.log_b1,
		log_b2 = params.log_b2)


	curves = \
		fit_fingerprints(
			task_params=task_params,
			model_params=model_params,
			train_params=train_params,
			verbose=params.verbose)

	with open(params.output_training_curve_fname, 'w') as f:
		f.write("train_loss,train_rmse,validation_rmse\n")
		[f.write("{},{},{}\n".format(train_loss, train_rmse, validation_rmse)) for train_loss,train_rmse, validation_rmse in zip(curves[0], curves[1], curves[2])]


if __name__ == '__main__':
	sys.exit(main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr))

