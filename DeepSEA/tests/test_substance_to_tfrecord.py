#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import tempfile
import numpy as np
import tensorflow as tf
from neuralfingerprint import load_data
from neuralfingerprint.mol_graph import graph_from_smiles, degrees, MolGraph
from neuralfingerprint.build_convnet import array_rep_from_smiles

#from DeepSEA.flat_substances import FlatSubstances

from DeepSEA.queue_substances import (
	prepare_substances_batch,
)

from DeepSEA.flat_substances import (
	FlatSubstances,
    smiles_to_flat_substance,
    serialize_flat_substance_example,
    parse_flat_substance_example,
	merge_flat_substances_network
)

class SubstancesToTFRecords(tf.test.TestCase):
	def test_nothing(self):
		pass


#    def test_merge_tf_flat_substance_atoms(self):
#        # Merge two sets each sorted by degree, so the resulting set
#        # is also sorted by degree. The strategy is to interleave the
#        # atoms based on the degree partitions for each set.
#
#        degrees = [0,1,2,3,4,5]
#
#        # atom_features for substance a and b
#        a = tf.get_variable(
#            name="a", initializer=tf.reshape(tf.range(33*62), [33, 62]))
#        b = tf.get_variable(
#            name="b", initializer=tf.reshape(tf.range(11*62) * 100, [11, 62]))
#
#        # partitions of a and b, computed from data (e.g. from the
#        # shape of the neighbor_atoms arrays)
#        n_a = tf.Variable([0, 5, 17, 9, 2, 0], name="partition_of_a") # always length 6
#        n_b = tf.Variable([0, 3,  5, 3, 0, 0], name="partition_of_a") # always length 6
#
#
#        indices_a = []
#        indices_b = []
#        count = tf.Variable(0, dtype=tf.int32, name="count")
#        for degree in degrees:
#           indices_a += [tf.range(count, count + n_a[degree])]
#           count += n_a[degree]
#           indices_b += [tf.range(count, count + n_b[degree])]
#           count += n_b[degree]
#
#        c = tf.dynamic_stitch(
#            [tf.concat(0,indices_a), tf.concat(0, indices_b)], [a, b], name="c")
#
#        with self.test_session() as sess:
#            sess.run(tf.initialize_all_variables())
#            x = c.eval()
#
#            # by the way we encoded the test data we can extract the atom_ids like this
#            atom_ids = x[:,0] / 62
#
#            expected_atom_ids = np.array([
#                # 5 degree 1 atoms from a:
#                0,  1,  2,  3,  4,
#                # 3 degree 1 atoms from b:
#                0, 100, 200,
#                # 17 degree 2 atoms from a:
#                5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
#                # 5 degree 2 atoms from b:
#                300, 400, 500, 600, 700,
#                # 9 degree 3 atoms from a:
#                22, 23, 24, 25, 26, 27, 28, 29, 30,
#                # 3 degree 3 atoms from b:
#                800, 900, 1000,
#                # 2 degree 4 atoms from a:
#                31, 32])
#            self.assertAllEqual(atom_ids, expected_atom_ids)
#
#	def test_relabel_neighors(self):
#		# three degree 3 atoms
#		#   0 -> [1,2,3]
#		#   1 -> [2,3,4]
#        #   2 -> [3,4,5]
#		atom_neighbors = tf.constant([[1,2,3], [2,3,4], [3,4,5]])
#
#		# relabel atoms
#		#   0 -> 10
#        #   1 -> 11
#        #   2 -> 12
#		#   3 -> 13
#        #   4 -> 14
#		#   5 -> 15
#		relabel = tf.constant([10, 11, 12, 13, 14, 15])
#
#		flat_atom_neighbors = tf.reshape(atom_neighbors, [-1])
#		flat_relabeled_atom_neighbors = tf.gather(params = relabel, indices = flat_atom_neighbors)
#		relabeled_atom_neighbors = tf.reshape(flat_relabeled_atom_neighbors, [-1, 3])
#
#		with self.test_session() as sess:
#			sess.run(tf.initialize_all_variables())
#			x = relabeled_atom_neighbors.eval()
#			self.assertAllEqual(np.array([[11, 12, 13], [12, 13, 14], [13, 14, 15]]), x)
#
#
#	def test_relabeling_bonds(self):
#		smiles1 = 'c1ccccc1'
#		m1 = graph_from_smiles(smiles1)
#		expected_atom_bond_neighbor_list = [[0, 5], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
#		self.assertAllEqual(expected_atom_bond_neighbor_list, m1.neighbor_list('atom', 'bond'))
#
#
#		smiles2 = 'CC(=O)O'
#		m2 = graph_from_smiles(smiles2)
#		expected_atom_bond_neighbor_list = [[0], [0, 1, 2], [1], [2]]
#		self.assertAllEqual(expected_atom_bond_neighbor_list, m2.neighbor_list('atom', 'bond'))
#
#		big_graph = MolGraph()
#		big_graph.add_subgraph(m1)
#		big_graph.add_subgraph(m2)
#		big_graph.sort_nodes_by_degree('atom')
#		expected_atom_bond_neighbor_list = [[6], [7], [8]]
#		self.assertAllEqual(expected_atom_bond_neighbor_list, big_graph.neighbor_list(('atom', 1), 'bond'))
#
#		expected_atom_bond_neighbor_list = [[0, 5], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
#		self.assertAllEqual(expected_atom_bond_neighbor_list, big_graph.neighbor_list(('atom', 2), 'bond'))
#
#		expected_atom_bond_neighbor_list = [[6, 7, 8]]
#		self.assertAllEqual(expected_atom_bond_neighbor_list, big_graph.neighbor_list(('atom', 3), 'bond'))
#
#
#
#
#    def test_encode_boolean_features(self):
#        # I asked this question on stackoverflow:
#        # http://stackoverflow.com/questions/38443395/read-boolean-features-for-tensorflow
#
#        a = np.array([1,0,0,1,1,0], dtype=bool)
#        a_sparse = a.nonzero()[0].tolist()
#
#        sparse_bit_features = tf.train.Feature(int64_list = tf.train.Int64List(value= a_sparse))
#        features = {'sparse_bit_features' : sparse_bit_features }
#        example = tf.train.Example(features=tf.train.Features(feature=features))
#        serialized_example = example.SerializeToString()
#
#        # ...
#
#        with self.test_session() as sess:
#            sess.run(tf.initialize_all_variables())
#            features = {'sparse_bit_features' : tf.VarLenFeature(tf.int64)}
#            parsed_example = tf.parse_single_example(serialized_example, features)
#            sparse_bit_features = parsed_example['sparse_bit_features']
#            bit_features = tf.sparse_to_indicator(sparse_bit_features, vocab_size=6)
#            x = bit_features.eval()
#            self.assertAllEqual(x, a)
#
#            # how to cast to float32
#            x_float = tf.select(bit_features, tf.ones([6]), tf.zeros([6]))
#            weights = tf.constant([45.3, 2.2, 41.1, 33.2, 21.1, 1.0], dtype=tf.float32)
#            y = tf.mul(x_float, weights)
#            y_eval = y.eval()
#            expected_result = np.array([45.3, 0, 0, 33.2, 21.1, 0])
#            [self.assertAlmostEqual(expected_result[i], y_eval[i], places=3) for i in range(6)]
#
#
#    def test_flat_substances_init(self):
#
#		FlatSubstances(
#		    substance_ids = np.array(["sub1"], dtype=str),
#		    substance_atoms = np.array([[0,0], [0,1]], dtype=np.int32),
#		    atom_features = np.array([[0,1,1,1], [1,0,0,0]], dtype=bool),
#		    bond_features = np.array([[0,1,1,1,1,1,1]], dtype=bool),
#		    atom_neighbors = np.array([[1], [0]], dtype=np.int32),
#		    bond_neighbors = np.array([[0], [0]], dtype=np.int32))
#
#    def test_flat_substance_representation(self):
#        substance_id = "domaine"
#        smiles = "NCCc1ccc(O)c(O)c1"
#		nf_flat_substance = array_rep_from_smiles((smiles,))
#        flat_substance = smiles_to_flat_substance(substance_id, smiles)
#
#		self.assertAllEqual(nf_flat_substance['atom_features'], flat_substance.atom_features)
#		self.assertAllEqual(nf_flat_substance['bond_features'], flat_substance.bond_features)
#		self.assertAllEqual(nf_flat_substance['atom_list'], flat_substance.substance_atoms)
#		for degree in degrees:
#			self.assertAllEqual(
#				nf_flat_substance[('atom_neighbors', degree)],
#				flat_substance.atom_neighbors[degree])
#			self.assertAllEqual(
#				nf_flat_substance[('bond_neighbors', degree)],
#				flat_substance.bond_neighbors[degree])
#
#	def test_flat_substance_encoding_decoding(self):
#        substance_id = "dopamine"
#        smiles = "NCCc1ccc(O)c(O)c1"
#
#        pre_flat_substance = smiles_to_flat_substance(substance_id, smiles)
#
#        label = 9.2 # pKi at DRD2
#        serialized_flat_substance_example = serialize_flat_substance_example(
#			pre_flat_substance, label)
#
#        with self.test_session() as sess:
#            sess.run(tf.initialize_all_variables())
#            parsed_example = parse_flat_substance_example(serialized_flat_substance_example)
#            post_flat_substance = FlatSubstances(
#				substance_ids = parsed_example['substance_ids'].eval(),
#				substance_atoms = parsed_example['substance_atoms'].eval(),
#				atom_features = parsed_example['atom_features'].eval(),
#				bond_features = parsed_example['bond_features'].eval(),
#				atom_neighbors = [
#					parsed_example['atom_neighbors_{}'.format(degree)].eval() for
#					degree in degrees],
#				bond_neighbors = [
#					parsed_example['bond_neighbors_{}'.format(degree)].eval() for
#					degree in degrees])
#
#			self.assertAllEqual(
#				pre_flat_substance.substance_ids, post_flat_substance.substance_ids)
#			self.assertAllEqual(
#				pre_flat_substance.substance_atoms, post_flat_substance.substance_atoms.indices)
#			self.assertAllEqual(pre_flat_substance.atom_features, post_flat_substance.atom_features)
#			self.assertAllEqual(pre_flat_substance.bond_features, post_flat_substance.bond_features)
#			for degree in degrees:
#				self.assertAllEqual(
#					pre_flat_substance.atom_neighbors[degree],
#					post_flat_substance.atom_neighbors[degree].tolist())
#				self.assertAllEqual(
#					pre_flat_substance.bond_neighbors[degree],
#					post_flat_substance.bond_neighbors[degree].tolist())
#
#
#	def test_flat_subtances_batch(self):
#        substance_id1 = "dopamine"
#        smiles1 = "NCCc1ccc(O)c(O)c1"
#        label1 = 9.2 # pKi at DRD2
#
#        substance_id2 = "serotonin"
#        smiles2 = "NCCc1c[nH]c2ccc(O)cc12"
#        label2 = 10.05 # pKi at 5HT2C
#
#		array_rep = array_rep_from_smiles((smiles1, smiles2,))
#		pre_flat_substances = FlatSubstances(
#			substance_ids = [substance_id1, substance_id2],
#			substance_atoms = array_rep['atom_list'],
#			atom_features = array_rep['atom_features'],
#			bond_features = array_rep['bond_features'],
#			atom_neighbors = [array_rep[('atom_neighbors', d)] for d in degrees],
#			bond_neighbors = [array_rep[('bond_neighbors', d)] for d in degrees])
#
#		example_queue = tf.train.string_input_producer(
#			string_tensor=[
#				serialize_flat_substance_example(
#					smiles_to_flat_substance(substance_id1, smiles1), label2),
#				serialize_flat_substance_example(
#					smiles_to_flat_substance(substance_id2, smiles2), label2)],
#			num_epochs=1, shuffle=False, seed=None, capacity=2)
#
#        with self.test_session() as sess:
#            sess.run(tf.initialize_all_variables())
#
#			batch_size = 2
#			parsed_examples = []
#			for i in range(batch_size):
#				# I think typically this should be in  a try/catch block to see if the queue is empty
#				example = example_queue.dequeue()
#				parsed_example = parse_flat_substance_example(example)
#				parsed_examples.append(parsed_example)
#
#			post_flat_substances = merge_flat_substances_network(parsed_examples)
#
#			self.assertAllEqual(
#				pre_flat_substance.substance_ids, post_flat_substance.substance_ids.eval())
#			self.assertAllEqual(
#				pre_flat_substance.substance_atoms,
#				post_flat_substance.substance_atoms.eval().indices)
#			self.assertAllEqual(
#				pre_flat_substance.atom_features,
#				post_flat_substance.atom_features.eval())
#			self.assertAllEqual(
#				pre_flat_substance.bond_features,
#				post_flat_substance.bond_features.eval())
#			for degree in degrees:
#				self.assertAllEqual(
#					pre_flat_substance.atom_neighbors[degree],
#					post_flat_substance.atom_neighbors[degree].eval().tolist())
#				self.assertAllEqual(
#					pre_flat_substance.bond_neighbors[degree],
#					post_flat_substance.bond_neighbors[degree].eval().tolist())

	def test_flat_subtances_batch(self):
		task_params = dict(
			substances_fnames = ['tests/test_data/substances.tsv'],
			field_delim="\t")
		model_params = dict(
			batch_size=2)

		substances_batch, labels_batch = prepare_substances_batch(model_params, task_params)

        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            sess.run(tf.initialize_local_variables())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			self.assertFalse(coord.should_stop())

			substances_batch_eval, labels_batch_eval = sess.run([substances_batch, labels_batch])

			import pdb
			pdb.set_trace()

			self.assertAllEqual(substances_batch_eval[0].shape, (22, 62))#[n_atoms, n_atom_features]
			self.assertAllEqual(substances_batch_eval[1].shape, (22, 6)) #[n_bonds, n_bond_features]
			self.assertAllEqual(substances_batch_eval[2].shape, (11, 2)) #[n_atoms, 2]
			self.assertAllEqual(substances_batch_eval[3].shape, (0,))    #[n_degree_0_atoms, 0]
			self.assertAllEqual(substances_batch_eval[4].shape, (6,1))   #[n_degree_1_atoms, 1]
			self.assertAllEqual(substances_batch_eval[5].shape, (10,2))  #[n_degree_2_atoms, 2]
			self.assertAllEqual(substances_batch_eval[6].shape, (6,3))   #[n_degree_3_atoms, 3]
			self.assertAllEqual(substances_batch_eval[7].shape, (0,))    #[n_degree_4_atoms, 4]
			self.assertAllEqual(substances_batch_eval[8].shape, (0,))    #[n_degree_5_atoms, 5]
			self.assertAllEqual(substances_batch_eval[9].shape, (0,))    #[n_degree_0_atoms, 0]
			self.assertAllEqual(substances_batch_eval[10].shape, (6,1))  #[n_degree_1_atoms, 1]
			self.assertAllEqual(substances_batch_eval[11].shape, (10,2)) #[n_degree_2_atoms, 2]
			self.assertAllEqual(substances_batch_eval[12].shape, (6,3))  #[n_degree_3_atoms, 3]
			self.assertAllEqual(substances_batch_eval[13].shape, (0,))   #[n_degree_4_atoms, 4]
			self.assertAllEqual(substances_batch_eval[14].shape, (0,))   #[n_degree_5_atoms, 5]


if __name__ == '__main__':
    unittest.main()
