#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves import xrange  # pylint: disable=redefined-builtin

from collections import namedtuple
import numpy as np
import pandas as pd
import tensorflow as tf
from neuralfingerprint.features import num_atom_features
from neuralfingerprint.features import num_bond_features
from neuralfingerprint.mol_graph import graph_from_smiles, degrees
NUM_ATOM_FEATURES = num_atom_features()
NUM_BOND_FEATURES = num_bond_features()

"""

  substance_data_to_tfrecods(task_params)
	flat_substance = smiles_to_flat_substance(smiles)
	serialized_flat_substance_example = serialize_flat_substance_example(flat_substance, label)

	write.write(serialized_flat_substance_example)
	_, serialized_flat_substance_example = reader.read(fname_queue)


  flat_substance, label = parse_flat_substance_example(serialized_flat_substance_example)

  flat_substance_set, label_batch = shuffle_batch(flat_substance)
  flat_mol_batch = batch_flat_mols(flat_mol_set)


"""

FlatSubstances = namedtuple(
	typename="FlatSubstances",
	field_names=[
		'substance_ids',
		'substance_atoms',
		'atom_features',
		'bond_features',
		'atom_neighbors',
		'bond_neighbors'])

"""
A FlatSubstances is a compact array-based representation for a set of molecules.

Each substance is represented as a directed graph
with features on each atom (node) and bond (edge).

The graph topology is stored by sorting the nodes by degree and
constructing neighbor_atom and neighbor_bond lists for each node.

substance_ids: np.array(shape=[n_substances], dtype=str) identifiers of for each substance
substance_atoms: np.array(shape=[2, n_atoms], dtype=int32) map from substance to atom
atom_features: np.array(shape=[n_atoms, n_atom_features], dtype=bool)
   features of each atom
   atoms are sorted by degree
bond_features: np.array(shape=[n_bonds, n_bond_features], dtype=bool)
   feature of each bond
atom_neighbors: [np.array(shape[n_atoms_with_degree_d, d], dtype=int32) for d in degrees]
   neighbors of each atom, grouped by degree
bond_neighbors: [np.array(shape[n_atoms_with_degree_d, d], dtype=int32) for d in degrees]
   bonds connecting to neighbors of each atom, grouped by degree

"""

def smiles_to_flat_substance(substance_id, smiles):
	"""
	Take a substance_id and smiles and turn it into an instance of a FlatSubstances
	"""
	substance_graph = graph_from_smiles(smiles)
	substance_graph.sort_nodes_by_degree('atom')

	substance_atoms = substance_graph.neighbor_list('molecule', 'atom')
	substance_atoms = np.array(
		[[[substance_i, atom_ij] for atom_ij in atoms_i]
        for substance_i, atoms_i in enumerate(substance_atoms)][0])
	flat_substance = FlatSubstances(
		substance_ids = [substance_id],
		substance_atoms = substance_atoms,
		atom_features = substance_graph.feature_array('atom'),
		bond_features = substance_graph.feature_array('bond'),
		atom_neighbors = [ substance_graph.neighbor_list(('atom', d), 'atom') for d in degrees ],
		bond_neighbors = [ substance_graph.neighbor_list(('atom', d), 'bond') for d in degrees ])
	return flat_substance

def serialize_flat_substance_example(flat_substance, label):
	"""
	transform a FlatSubstance into a tf.train.Example with features

	"""

	feature = {
		'label': tf.train.Feature(float_list = tf.train.FloatList(value=[label])),
		'substance_ids': tf.train.Feature(
			bytes_list = tf.train.BytesList(
				value=[id.encode('utf-8') for id in flat_substance.substance_ids])),
		'n_atoms': tf.train.Feature(
		    int64_list = tf.train.Int64List(
				value=[len(atoms) for atoms in flat_substance.atom_neighbors])),
		'n_bonds': tf.train.Feature(
		    int64_list = tf.train.Int64List(
				value=[flat_substance.atom_features.shape[0]])),
		'substance_atoms': tf.train.Feature(
		    int64_list = tf.train.Int64List(
				value=flat_substance.substance_atoms.reshape(-1).tolist())),
		'atom_features': tf.train.Feature(
			int64_list = tf.train.Int64List(
				value=flat_substance.atom_features.reshape(-1).nonzero()[0].tolist())),
		'bond_features': tf.train.Feature(
			int64_list = tf.train.Int64List(
				value=flat_substance.bond_features.reshape(-1).nonzero()[0].tolist()))}
	for d, atom_neighbors in enumerate(flat_substance.atom_neighbors):
		feature['atom_neighbors_{}'.format(d)] = tf.train.Feature(
			int64_list = tf.train.Int64List(value=np.reshape(atom_neighbors, [-1])))

	for d, bond_neighbors in enumerate(flat_substance.bond_neighbors):
		feature['bond_neighbors_{}'.format(d)] = tf.train.Feature(
			int64_list = tf.train.Int64List(value=np.reshape(bond_neighbors, [-1])))

	flat_substance_example = tf.train.Example(features = tf.train.Features(feature=feature))
	serialized_flat_substance_example = flat_substance_example.SerializeToString()

	return serialized_flat_substance_example

def substance_data_to_tfrecods(task_params, verbose=True):
	"""
	Convert .csv file to tfrecods file as input for tensorflow network

	task_params is a dict having the following keys
		input_data_fname: .csv file with specified columns
		substance_id_column: identifier interpreted as a string
		smiles_column: substance in smiles format
		label_column: label (e.g. activity or property) of the smiles

	The whole .csv is read into memory and the rows are randomized.

	"""
	if verbose:
		print("Reading substances from '{}':\n\tsubstance_id column: '{}'\n\tsmiles column: '{}'\n\tlabel column: '{}'".format(
			task_params['input_data_fname'],
			task_params['substance_id_column'],
			task_params['smiles_column'],
			task_params['label_column']))

		print("Writing tfrecods to '{}'".format(task_params["output_data_fname"]))

	df = pd.read_csv(task_params['input_data_fname'])
	df.reindex(np.random.permutation(df.index))

	with tf.python_io.TFRecordWriter(task_params["output_data_fname"]) as writer:
		for index, row in df.iterrows():
			if index % 500 == 0 and verbose: print("Reading row: {} ...".format(index))

			substance_id = getattr(row, task_param['substance_id_column'])
			smiles = getattr(row, task_params['smiles_column'])
			label = getattr(row, task_params['label_column'])
			flat_substance = smiles_to_flat_substance(smiles)
			serialized_example = serialize_flat_substance_example(flat_substance, label)
			example_serialized = smiles_label_to_tfrecord(smiles, label)
			writer.write(example_serialized)


def parse_flat_substance_example(serialized_example):
	features={
		'substance_ids': tf.VarLenFeature(tf.string),
		'label': tf.FixedLenFeature([1], tf.float32),
		'n_atoms' : tf.FixedLenFeature([len(degrees)], tf.int64),
		'n_bonds' : tf.FixedLenFeature([1], tf.int64),
		'substance_atoms': tf.VarLenFeature(tf.int64),
		'atom_features': tf.VarLenFeature(tf.int64),
		'bond_features': tf.VarLenFeature(tf.int64)}
	for degree in degrees:
		features["atom_neighbors_{}".format(degree)] = tf.VarLenFeature(tf.int64)
		features["bond_neighbors_{}".format(degree)] = tf.VarLenFeature(tf.int64)

	example = tf.parse_single_example(serialized_example, features)

	substance_ids = example['substance_ids'].values
	substance_ids.set_shape([1])
	example['substance_ids'] = substance_ids

	n_substances = tf.shape(substance_ids)[0]   # int32
	n_atoms = tf.reduce_sum(example['n_atoms']) # int64
	n_bonds = example['n_bonds'][0]             # int64

	substance_atoms_indices = example['substance_atoms'].values
	substance_atoms_indices = tf.reshape(substance_atoms_indices, [-1, 2])
	new_values = tf.fill(tf.expand_dims(tf.to_int32(n_atoms), 0), True)
	substance_atoms = tf.SparseTensor(
		substance_atoms_indices, new_values, [tf.to_int64(n_substances), n_atoms])
	example['substance_atoms'] = substance_atoms

	atom_features = example['atom_features']
	atom_features = tf.sparse_to_indicator(
		atom_features,
		vocab_size=tf.to_int32(n_atoms)*NUM_ATOM_FEATURES)
	atom_features = tf.reshape(atom_features, [tf.to_int32(n_atoms), NUM_ATOM_FEATURES])
	example['atom_features'] = atom_features

	bond_features = example['bond_features']
	bond_features = tf.sparse_to_indicator(bond_features, vocab_size=n_bonds*NUM_BOND_FEATURES)
	bond_features = tf.reshape(bond_features, [tf.to_int32(n_bonds), NUM_BOND_FEATURES])
	example['bond_features'] = bond_features

	for degree in degrees:
		shape= tf.to_int32(tf.pack([example['n_atoms'][degree], degree]))
		atom_neighbors = example["atom_neighbors_{}".format(degree)]
		atom_neighbors = tf.sparse_tensor_to_dense(atom_neighbors)
		atom_neighbors = tf.reshape(atom_neighbors, shape)
		example["atom_neighbors_{}".format(degree)] = atom_neighbors

		bond_neighbors = example["bond_neighbors_{}".format(degree)]
		bond_neighbors = tf.sparse_tensor_to_dense(bond_neighbors)
		bond_neighbors = tf.reshape(bond_neighbors, shape)
		example["bond_neighbors_{}".format(degree)] = bond_neighbors

	return example



#	batch = tf.train.shuffle_batch(
#		example,
#		batch_size=model_params['batch_size'],
#		capacity=2000,
#		min_after_dequeue=1000)


def merge_flat_substances_network(batch):

	import pdb
	pdb.set_trace()

	# Interleave atoms features to maintain degree sort order
	atom_indices = []
	atom_count = tf.constant(0, dtype=tf.int32, name="merged_atom_count")
	for degree in degrees:
		for i, substance in enumerate(batch):
		    n_atoms = substance['atom_neighbors_{}'.format(degree)].get_shape()[0]
		    atom_indices[i] += [tf.range(atom_count, atom_count + n_atoms)]
		    tf.assign_add(atom_count, n_atoms)

	atom_features = tf.dynamic_stitch(
		[tf.concat(0, substance_atom_indicies) for substance_atom_indicies in atom_indices],
		[substance['atom_features'] for substance in batch],
		name="merged_atom_features")

	atom_neighbors = []
	for degree in degrees:
		atom_neighbors_by_degree = []
		for i, substance in enumerate(batch):
		    atom_neighbors = tf.reshape(substance['atom_neighbors_{}'.format(degree)], [-1])
		    relabeled_atom_neighbors = tf.gather(atom_indices[i], atom_neighbors)
		    relabeled_atom_neighbors = tf.reshape(relabeled_atom_neighbors, [-1, degree])
		    atom_neighbors_by_degree += [relabeled_atom_neighbors]
		atom_neighbors.append(tf.concat(0, atom_neighbors_by_degree))


	substance_atoms = tf.SparseTensor(
		indices=tf.pack(atom_indices),
		values=[atom_count],
		shape=[len(batch), atom_count])

	bond_indices = []
	bond_count = tf.contsant(0, dtype=tf.int32, name="merged_bond_count")
	for i, substance in enumerate(batch):
		n_bonds = substance['bond_features'].get_shape()[0]
		bond_indices[i] = [tf.range(bond_count, bond_count + n_atoms)]
		tf.assign_add(bond_count, n_bonds)

	bond_features = tf.concat(0, [substance['bond_features'] for substance in batch])

	bond_neighbors = []
	for degree in degrees:
		bond_neighbors_by_degree = []
		for i, substance in enumerate(batch):
			bond_neighbors = tf.reshape(subtance["bond_neighbors_{}".format(degree)], [-1])
		    relabeled_bond_neighbors = tf.gather(bond_indices[i], bond_neighbors)
		    relabeled_bond_neighbors = tf.reshape(relabeled_bond_neighbors, [-1, degree])
		    bond_neighbors_by_degree += [relabeled_bond_neighbors]
		bond_neighbors.append(tf.gather(0, bond_neighbors_by_degree))

	return merged_batch
