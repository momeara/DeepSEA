#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from cPicle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
from mol_graph import degrees


def read_single_example(tfrecords_fname):
    """
    Each time this is called, it reads an example from tfrecords
    where each example has features 'label':str and 'graph':MolGraph.

    Must be run in the context of a tensorflow graph.
    
    """

    fname_queue = tf.train.string_input_producer([tfrecords_fname], num_epochs=None)
    reader = tf.TFRecordReader()
    
    _, serialized_example = reader.read(fname_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'graph': tf.VarLenFeature(tf.string())
        })

    label = features['label']
    graph = pickle.load(features['graph'])
    return label, graph


def batch_examples_network(label, graph, model_params):

    labels_batch, graphs_batch = tf.train.shuffle_batch(
        [label, graph],
        batch_size=model_params['batch_size'],
        capacity=2000,
        min_after_dequeue=1000)

    molgraph = MolGraph()
    for subgraph in graphs_batch:
        molgraph.add_subgraph(subgraph)

    molgraph.sort_nodes_by_degree('atom')

    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return labels_batch, arrayrep
