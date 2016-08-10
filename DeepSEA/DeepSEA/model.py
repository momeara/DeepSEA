#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:


# adapt neural-fingeprint to tensorflow
# to fit fingerprints based on a single endpoint regression


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
from neuralfingerprint.features import num_atom_features, num_bond_features
from neuralfingerprint.mol_graph import degrees

from DeepSEA.rdkit_util import (
	smiles_to_fps,
)


def _add_weights(variables, weight_key, shape, train_params, op=tf.random_normal):
	weights = tf.Variable(
		initial_value=op(shape, stddev=np.exp(train_params['log_init_scale'])),
		name=weight_key)
	variables[weight_key] = weights
	with tf.name_scope("regularization/") as regularization_scope:
		variables['l2_loss'] += tf.nn.l2_loss(weights)
		variables['l1_loss'] += tf.reduce_sum(tf.abs(weights))
		# to be consistent with neuralfingerprint, the
		# regularization the mean of squares or absolute values
		# rather than the sum. So we need to keep track of how
		# many variables there are to divide by before applying it
		# to the loss.
		variables['n_params'] += np.prod(shape)


def initialize_fingerprint_variables(train_params, model_params):
	variables = {}
	with tf.name_scope("regularization/") as scope:
		variables['l2_loss'] = tf.constant(0.0, name="l2_loss")
		variables['l1_loss'] = tf.constant(0.0, name="l1_loss")
		variables['n_params'] = 0

	if model_params['fp_type'] == 'neural':
		with tf.name_scope("neural_fingerprint") as scope:
			# neural fingerprint layer output weights and biases
			all_layer_sizes = [num_atom_features()] + [model_params['fp_width']] * model_params['fp_depth']
			for layer in range(len(all_layer_sizes)):
				with tf.name_scope("layer_{}".format(layer)):
					_add_weights(
						variables = variables,
						weight_key = 'layer_output_weights_{}'.format(layer),
						shape = [all_layer_sizes[layer], model_params['fp_length']],
						train_params = train_params)

					_add_weights(
						variables = variables,
						weight_key = 'layer_output_bias_{}'.format(layer),
						shape = [model_params['fp_length']],
						train_params = train_params)

			 # neural fingerprint graph integration layer weights and biases
			in_and_out_sizes = zip(all_layer_sizes[:-1], all_layer_sizes[1:])
			for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
				with tf.name_scope("layer_{}/".format(layer)) as layer_scope:
					_add_weights(
						variables = variables,
						weight_key = "layer_{}_biases".format(layer),
						shape = [N_cur],
						train_params = train_params)

					_add_weights(
						variables = variables,
						weight_key = "layer_{}_self_filter".format(layer),
						shape = [N_prev, N_cur],
						train_params = train_params)

					for degree in degrees:
						_add_weights(
							variables = variables,
							weight_key = 'layer_{}_neighbor_{}_filter'.format(layer, degree),
							shape = [N_prev + num_bond_features(), N_cur],
							train_params = train_params)
	return variables


def initialize_convolution_prediction_variables(train_params, model_params):
	variables = {}
	with tf.name_scope("regularization/") as scope:
		variables['l2_loss'] = tf.constant(0.0, name="l2_loss")
		variables['l1_loss'] = tf.constant(0.0, name="l1_loss")
		variables['n_params'] = 0

	with tf.name_scope("convolution_prediction") as scope:
		# prediction network weights and biases
		layer_sizes = model_params['prediction_layer_sizes'] + [1]
		for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			_add_weights(
				variables = variables,
				weight_key = "prediction_weights_{}".format(i),
				shape = shape,
				train_params = train_params)

			_add_weights(
				variables = variables,
				weight_key = "prediction_biases_{}".format(i),
				shape = [shape[1]],
				train_params = train_params)

	return variables

def initialize_linear_regression_prediction_variables(train_params, model_params):
	variables = {}
	with tf.name_scope("regularization/") as scope:
		variables['l2_loss'] = tf.constant(0.0, name="l2_loss")
		variables['l1_loss'] = tf.constant(0.0, name="l1_loss")
		variables['n_params'] = 0

	with tf.name_scope("linear_regression_prediction") as scope:
		# prediction network weights and biases
		_add_weights(
			variables = variables,
			weight_key = "linear_regression_weights".format(i),
			shape = [model_params["fp_length"]],
			train_params = train_params)

	return variables



def load_variables(
	fp_parser, fp_weights,
	net_parser, net_weights,
	fingerprint_variables,
	prediction_variables,
	model_params):

	num_hidden_features = [model_params['fp_width']] * model_params['fp_depth']
	for layer in xrange(len(num_hidden_features)):
		variables["layer_output_weights_{}".format(layer)] = tf.assign(
			variables["layer_output_weights_{}".format(layer)],
			fp_parser.get(fp_weights, ("layer output weights", layer)))
		variables["layer_output_bias_{}".format(layer)] = tf.assign(
			variables["layer_output_bias_{}".format(layer)],
			tf.to_float(tf.squeeze(fp_parser.get(fp_weights, ("layer output bias", layer)))))

		variables["layer_{}_biases".format(layer)] = tf.assign(
			variables["layer_{}_biases".format(layer)],
			tf.to_float(tf.squeeze(fp_parser.get(fp_weights, ("layer", layer, "biases")))))
		variables["layer_{}_self_filter".format(layer)] = tf.assign(
			variables["layer_{}_self_filter".format(layer)],
			fp_parser.get(fp_weights, ("layer", layer, "self filter")))

		for degree in degrees:
			variables["layer_{}_neighbor_{}_filter".format(layer, degree)] = tf.assign(
				variables["layer_{}_neighbor_{}_filter".format(layer, degree)],
				fp_parser.get(fp_weights, ("layer {} degree {} filter".format(layer, degree))))

		layer_sizes = model_params['prediction_layer_sizes'] + [1]
		for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
			variables["prediction_weights_{}".format(i)] = tf.assign(
				variables["prediction_weights_{}".format(i)],
				net_parser.get(net_weights, ("weights", i)))

			variables["prediction_biases_{}".format(i)] = tf.assign(
				variables["prediction_biases_{}".format(i)],
				tf.to_float(tf.squeeze(net_parser.get(net_weights, ("biases", i)), squeeze_dims=[0])))




def build_fingerprint_summary_network(fps, loss, fingerprint_variables, model_params):
	tf.histogram_summary("fingerprints", fps)

	def max_n(inputs):
		m = [tf.reshape(tf.reduce_max(variable), [-1]) for variable in inputs]
		return tf.reduce_max(tf.concat(0, m))

	def mean_n(inputs):
		m = [tf.reshape(variable, [-1]) for variable in inputs]
		return tf.reduce_mean(tf.concat(0, m))

	tf.scalar_summary("max_fingerprint_entry", max_n([fps]))
	tf.scalar_summary("mean_fingerprint_entry", mean_n([fps]))


	tf.scalar_summary("max_weight", max_n(tf.trainable_variables()))
	tf.scalar_summary("mean_weight", mean_n(tf.trainable_variables()))

	num_hidden_features = [model_params['fp_width']] * model_params['fp_depth']
	tf.scalar_summary("max_layer_output_weights",
		max_n([fingerprint_variables["layer_output_weights_{}".format(layer)]
			for layer in xrange(len(num_hidden_features))]))
	tf.scalar_summary("mean_layer_output_weights",
		mean_n([fingerprint_variables["layer_output_weights_{}".format(layer)]
			for layer in xrange(len(num_hidden_features))]))
	tf.scalar_summary("max_layer_output_bias",
		max_n([fingerprint_variables["layer_output_bias_{}".format(layer)]
			for layer in xrange(len(num_hidden_features))]))
	tf.scalar_summary("mean_layer_output_bias",
		mean_n([fingerprint_variables["layer_output_bias_{}".format(layer)]
			for layer in xrange(len(num_hidden_features))]))

	tf.scalar_summary("max_layer_biases",
		max_n([fingerprint_variables["layer_{}_biases".format(layer)]
			for layer in xrange(len(num_hidden_features))]))
	tf.scalar_summary("mean_layer_biases",
		mean_n([fingerprint_variables["layer_{}_biases".format(layer)]
			for layer in xrange(len(num_hidden_features))]))

	tf.scalar_summary("max_layer_{}_self_filter",
		max_n([fingerprint_variables["layer_{}_self_filter".format(layer)]
			for layer in xrange(len(num_hidden_features))]))
	tf.scalar_summary("mean_layer_self_filter",
		mean_n([fingerprint_variables["layer_{}_self_filter".format(layer)]
			for layer in xrange(len(num_hidden_features))]))

def build_prediction_summary_network(fps, loss, prediction_variables, model_params):
	def max_n(inputs):
		m = [tf.reshape(tf.reduce_max(variable), [-1]) for variable in inputs]
		return tf.reduce_max(tf.concat(0, m))

	def mean_n(inputs):
		m = [tf.reshape(variable, [-1]) for variable in inputs]
		return tf.reduce_mean(tf.concat(0, m))


	tf.scalar_summary("max_neighbor_filter",
		max_n([variables["layer_{}_neighbor_{}_filter".format(layer, degree)]
			for layer in xrange(len(num_hidden_features))
			for degree in degrees]))
	tf.scalar_summary("mean_neighbor_filter",
		mean_n([variables["layer_{}_neighbor_{}_filter".format(layer, degree)]
			for layer in xrange(len(num_hidden_features))
			for degree in degrees]))

	tf.scalar_summary("loss", loss)


def build_neural_fps_network(substances, variables, model_params):

	def matmult_neighbors(atom_features, layer, substances, variables):
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

		neighbor_features: [
		    [N_atoms with degree, degree, num_atom_features],
		    [N_atoms with degree, degree, num_bond_features] ]

		stacked_neigbors: [N_atoms with degree, degree, num_atom_features + num_bond_features]
		summed_neighbors: [N_atoms with degree, num_atom_features + num_bond_features]
		activations: [N_atoms with degree, num_atom_features]
		return [N_atoms, num_atom_features]

		"""
		with tf.name_scope("matmul_neighbors/") as matmul_neighbors_scope:
			activations_by_degree = []
			for degree in degrees:
				atom_neighbor_list = substances['atom_neighbors_{}'.format(degree)]
				bond_neighbor_list = substances['bond_neighbors_{}'.format(degree)]
				neighbor_features = [
					tf.gather(params=atom_features, indices=atom_neighbor_list),
					tf.gather(params=substances['bond_features'], indices=bond_neighbor_list)]
				stacked_neighbors = tf.concat(concat_dim=2, values=neighbor_features)
				summed_neighbors = tf.reduce_sum(stacked_neighbors, reduction_indices=1)
				neighbor_filter = variables['layer_{}_neighbor_{}_filter'.format(layer, degree)]
				activations = tf.matmul(summed_neighbors, neighbor_filter)
				activations_by_degree.append(activations)
				activations = tf.concat(
					concat_dim=0, values=activations_by_degree, name="activations")
			return activations

	def update_layer(atom_features, layer, substances, variables):
		with tf.name_scope("layer_{}/".format(layer)) as update_layer_scope:
			layer_bias		= variables["layer_{}_biases".format(layer)]
			layer_self_filter = variables["layer_{}_self_filter".format(layer)]
			self_activations = tf.matmul(atom_features, layer_self_filter)
			neighbor_activations = matmult_neighbors(
				atom_features, layer, substances, variables)
			activations = tf.nn.bias_add(tf.add(neighbor_activations, self_activations), layer_bias)
			activations_mean, activations_variance = tf.nn.moments(activations, [0], keep_dims=True)
			# batch normalization a la neural fingerprints
			activations = (activations - activations_mean) / (tf.sqrt(activations_variance) + 1)

			#activations = tf.nn.batch_normalization(
			#	activations, activations_mean, activations_variance,
			#	offset=None, scale=None, variance_epsilon=1e-3)
			activations = tf.nn.relu(activations, name="activations")
			return activations

	def write_to_fingerprint(atom_features, layer, substances, variables):
		"""
		N_atoms = 14700 (for example)
		N_compounds = 800 (for example)
		num_atom_features = 20

		atom_features: [N_atoms, num_atom_features]
        substance_atoms: SparseTensor[N_substances, N_atoms]

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
				substances['substance_atoms'], atom_outputs, name=scope)
			return layer_output

	with tf.name_scope("fingerprint/") as fingerprint_scope:
		atom_features = substances['atom_features']
		fps = write_to_fingerprint(atom_features, 0, substances, variables)

		num_hidden_features = [model_params['fp_width']] * model_params['fp_depth']
		for layer in xrange(len(num_hidden_features)):
			atom_features = update_layer(atom_features, layer, substances, variables)
			fps += write_to_fingerprint(atom_features, layer+1, substances, variables)

		return fps

def build_morgan_fps_network(smiles, eval_params, model_params):

	def func(smiles):
		return smiles_to_fps(smiles, model_params['fp_radius'], model_params['fp_length'])

	morgan_fps_list = tf.py_func(
		func=func,
		inp=[smiles],
		Tout=[tf.float32],
		name="RDKit_morgan_fingerprint")

	morgan_fps = morgan_fps_list[0]
	morgan_fps.set_shape([eval_params['batch_size'], model_params['fp_length']])
	return morgan_fps

def build_continuous_jaccard_distance_network(fps1, fps2):
	"""
	Given two batches of fingerprints, compute the distance between
	the i'th fingerprint from each batch, using the continuous
	generalization of the Jaccard distance:

	    1 - \Sum(min(x_i, y_i)) / \Sum(max(x_i, y_i))

	see (Duvenaud, NIPS 2015)

	"""
	intersect = tf.reduce_sum(tf.minimum(fps1, fps2), [1], name="intersect")
	union = tf.reduce_sum(tf.maximum(fps1, fps2), [1], name="union")
	tanimoto = tf.div(intersect, union, name="tanimoto")
	jaccard = tf.sub(1.0, tanimoto, name="jaccard")
	return jaccard


def build_convolution_prediction_network(fps, variables, model_params):

	"""
	Given a batch of fingerprints preduce a prediction for normalized numeric label
	using a vanilla convolutional network

    e.g.:
	batch_size = 100
	fp_width = 512
	prediction_layer_sizes = [512, 100, 1]

	fps = [batch_size, 512]
	weights_0 = [512, 100]
	biases_0 = [100]
	activations_0 = [batch_size, 100]

	hidden_1 = [batch_size, 100]
	weights_1 = [100, 1]
	biases_1 = [1]
	activations_1 = [batch_size, 1]

	"""
	with tf.name_scope("convolution_prediction") as convolution_prediction_scope:
		activations = fps
		layer_sizes = model_params['prediction_layer_sizes'] + [1]
		for layer in range(len(layer_sizes) - 1):
			weights = variables['prediction_weights_{}'.format(layer)]
			biases = variables['prediction_biases_{}'.format(layer)]
			activations = tf.nn.bias_add(tf.matmul(activations, weights), biases, name="activations")
			if layer < len(layer_sizes) - 2:
				activations_mean, activations_variance = tf.nn.moments(activations, [0], keep_dims=True)
				activations = (activations - activations_mean) / (tf.sqrt(activations_variance) + 1)
				activations = tf.nn.relu(activations)
		return tf.squeeze(activations, name=convolution_prediction_scope)

def build_linear_prediction_network(fps, variables, model_params):

	with tf.name_scope("linear_prediction") as scope:
		weights = variables['linear_prediction_weights']
		return tf.mul(fps, weights, name=scope)



def build_loss_network(
	normed_predictions,
	labels,
	fingerprint_variables,
	prediction_variables,
	model_params):

	with tf.name_scope("loss") as loss_scope:

		labels_mean, labels_variance = tf.nn.moments(
			labels, [0], keep_dims=False, name="label_moments")
		labels_std = tf.sqrt(labels_variance, name="labels_std")

		normed_labels = (labels - labels_mean) / labels_std

		# un-norm the normed-predictions
		predictions = tf.add(normed_predictions * labels_std, labels_mean, name="predictions")

		# http://stackoverflow.com/questions/33846069/how-to-set-rmse-cost-function-in-tensorflow
		# compute the rmse in the original space to get the units right
		rmse = tf.sqrt(tf.reduce_mean((normed_predictions - normed_labels)**2), name="rmse")

		fingerprint_regularization = tf.add(
			fingerprint_variables['l2_loss'] * model_params['l2_penalty'] / fingerprint_variables['n_params'],
			fingerprint_variables['l1_loss'] * model_params['l1_penalty'] / fingerprint_variables['n_params'],
			name="fingerprint_regularization")
		prediction_regularization = tf.add(
			prediction_variables['l2_loss'] * model_params['l2_penalty'] / prediction_variables['n_params'],
			prediction_variables['l1_loss'] * model_params['l1_penalty'] / prediction_variables['n_params'],
			name="prediction_regularization")
		regularization = tf.add(
			fingerprint_regularization, prediction_regularization,
			name="regularization")

		loss = tf.add(rmse, regularization, name=loss_scope)
		return predictions, loss


def build_triple_score_network(distance_to_plus, distance_to_minus, model_params):
	"""
	Compute the hinge loss triple loss from (Wang, CVPR2014)

	The hinge loss is a convex approximation to the 0-1 ranking error
    loss, which measures the model's violation of the ranking order
	specified in the triplet.

	The score_gap parameter favors a gap between the distance of the two image pairs
	"""

	triple_score = tf.nn.relu(tf.add(
		model_params['score_gap'],
		tf.sub(distance_to_plus, distance_to_minus)),
		name="triple_score")

	return triple_score

def build_triple_loss_network(triple_score, variables, model_params):
	fingerprint_regularization = tf.add(
		variables['l2_loss'] * model_params['l2_penalty'] / variables['n_params'],
		variables['l1_loss'] * model_params['l1_penalty'] / variables['n_params'],
		name="fingerprint_regularization")
	loss = tf.add(triple_score, fingerprint_regularization, name="loss")
	return loss


def build_optimizer(loss, train_params):
	with tf.name_scope("optimizer") as optimizer_scope:
		learning_rate = tf.constant(np.exp(train_params['log_learning_rate']))
		beta1 = tf.constant(np.exp(train_params['log_b1']))
		beta2 = tf.constant(np.exp(train_params['log_b2']))
		adam = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
		optimizer = adam.minimize(loss, name=optimizer_scope)
		return optimizer
