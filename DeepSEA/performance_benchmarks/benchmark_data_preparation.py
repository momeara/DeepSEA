#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

import os
from time import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import tensorflow as tf
from neuralfingerprint.mol_graph import graph_from_smiles

@contextmanager
def tictoc():
    print("--- Start clock ---")
    t1 = time()
    yield
    dt = time() - t1
    print("--- Stop clock: {} seconds elapsed ---".format(dt))


def prepare_data(input_fname, substance_id_column, smiles_column):
    print("prepare input data:")
    with tictoc():
        df = pd.read_csv(input_fname, sep='\t')
        data = []
        for index, row in df.iterrows():
            smiles = getattr(row, smiles_column)
            try:
                substance_graph = graph_from_smiles(smiles)
                data.append(substance_graph.feature_array('atom'))
            except Exception as e:
                substance_id = getattr(row, substance_id_column)
                print("failed to parse compound: {} with smiles {}. error: {}".format(substance_id, smiles, e.message))
    return data

def prepare_int64(data, output_fname):
    print("prepare int64:")
    with tictoc():
        with tf.python_io.TFRecordWriter(output_fname) as writer:
            for atom_features in data:
                value = atom_features.reshape(-1).tolist()
                bit_features = tf.train.Feature(int64_list =  tf.train.Int64List(value=value))
                features = {'bit_features' : bit_features}
                example = tf.train.Example(features=tf.train.Features(feature=features))
                serialized_example = example.SerializeToString()
                writer.write(serialized_example)
    print("File size: {}".format(os.path.getsize(output_fname)))

def prepare_sparse(data, output_fname):
    print("prepare sparse:")
    with tictoc():
        with tf.python_io.TFRecordWriter(output_fname) as writer:
            for atom_features in data:
                value = atom_features.reshape(-1).nonzero()[0].tolist()
                bit_features = tf.train.Feature(int64_list =  tf.train.Int64List(value=value))
                features = {'bit_features' : bit_features}
                example = tf.train.Example(features=tf.train.Features(feature=features))
                serialized_example = example.SerializeToString()
                writer.write(serialized_example)
    print("File size: {}".format(os.path.getsize(output_fname)))


def prepare_packed(data, output_fname):
    print("prepare packed:")
    with tictoc():
        with tf.python_io.TFRecordWriter(output_fname) as writer:
            for atom_features in data:
                value = np.packbits(atom_features.reshape(-1)).tobytes()
                bit_features = tf.train.Feature(bytes_list = tf.train.BytesList(value=value))
                features = {'bit_features' : bit_features}
                example = tf.train.Example(features=tf.train.Features(feature=features))
                serialized_example = example.SerializeToString()
                writer.write(serialized_example)
    print("File size: {} bytes".format(os.path.getsize(output_fname)))

if __name__ == '__main__':
    
    data = prepare_data("/mnt/nfs/work/momeara/sets/data_repo/inst/sets/chembl21/data/chembl_compounds.tsv", "zinc_id", "chembl_smiles")

#    prepare_int64(data, "/scratch/momeara/chembl_prep_int64.tfrecords")
    prepare_sparse(data, "/scratch/momeara/chembl_prep_sparse.tfrecords")
#    prepare_packed(data, "/scratch/momeara/chembl_prep_packed.tfrecords")
