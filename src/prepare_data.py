#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

# Load <cid> <smiles> <label> data into native tensorflow data format

# Consulted this resource
# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import cPickle as pickle
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import tensorflow as tf
from neuralfingerprint import load_data
from neuralfingerprint.mol_graph import graph_from_smiles



def prepare_data(task_params, verbose=False):

	df = pd.read_csv(task_params['input_data_fname'])
	df.reindex(np.random.permutation(df.index))

    with tf.python_io.TFRecordWriter(task_params["output_data_fname"]) as writer:
    	for index, row in df.iterrows():
			if index % 500 == 0: print("Reading row: {} ...".format(index))

    		smiles = getattr(row, task_params['smiles_column'])
    		label = getattr(row, task_params['target_column'])
            molgraph = graph_from_smiles(smiles)
			feature = {
				'label': tf.train.Feature(float_list = tf.train.FloatList(value=[label])),
				'atom_features': tf.train.Feature(
						int64_list = tf.train.Int64List(value=molgraph.feature_array('atom'))),
				'bond_features': tf.train.Feature(
						int64_list = tf.train.Int64List(value=molgraph.feature_array('bond')))}
			for degree in degrees:
				features['atom_neighbors_{}'.format(degree)] = tf.train.Feature(
					int64_list = tf.train.Int64List(
						value=np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)))
				features['bond_neighbors_{}'.format(degree)] = tf.train.Feature(
					int64_list = tf.train.Int64List(
						value=np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)))

            example = tf.train.Example(features = tf.train.Features(features))
            example_serialized = example.SerializeToString()
            writer.write(example_serialized)

def main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr):
    parser = ArgumentParser("Train a neural fingerprint function")

    #task_params
    parser.add_argument("--input_data_fname", help="Comma separated value file of substance activity data. After a header row, each row represents a substance and having columns identified by --smiles_column and --activity_column")
#    parser.add_argument("--output_fp_function_fname", help="Name of fingerprint function output file", default="fp_function.pickle")
    parser.add_argument("--output_data_fname", help="Name of output native data file e.g <input_data>.tfrecords")
    parser.add_argument("--smiles_column", help="Name of substance smiles column.", default="smiles")
    parser.add_argument("--target_column", help="Name of substance target column.", default="target")
    parser.add_argument("--verbose", default=False, action='store_true', help="Report verbose output")

    params, others = parser.parse_known_args(args)


    task_params = dict(
        input_data_fname = params.input_data_fname,
        output_data_fname = params.output_data_fname,
        smiles_column = params.smiles_column,
        target_column = params.target_column)

    prepare_data(task_params, verbose=params.verbose)

if __name__ == '__main__':
    sys.exit(main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr))

