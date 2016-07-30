#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
from neuralfingerprint.mol_graph import degrees
from neuralfingerprint.build_convnet import array_rep_from_smiles
from neuralfingerprint.features import (
    num_atom_features,
    num_bond_features,
)

n_atom_features = num_atom_features()
n_bond_features = num_bond_features()



def smiles_to_flat_substances_network(smiles, eval_params):
    """Prepare a batch of substances from smiles strings

    smiles should be a 1d tensor of smiles strings

    """
    def func(smiles):

        data = array_rep_from_smiles(tuple(smiles))

        # list of lists -> [n_atoms, 2] array with columns (substance_id, atom_id)
        substance_atoms = []
        for substance_i, atoms_i in enumerate(data['atom_list']):
            substance_atoms += [ [substance_i, atoms_ij] for atoms_ij in atoms_i]
        substance_atoms = np.array(substance_atoms)
        

        for degree in degrees:
            if data[('atom_neighbors', degree)].shape == (0,):
                data[('atom_neighbors', degree)].shape = (0,0)
            if data[('bond_neighbors', degree)].shape == (0,):
                data[('bond_neighbors', degree)].shape = (0,0)

        return [
            substance_atoms.shape[0],    
            data['atom_features'].astype(np.float32),
            data['bond_features'].astype(np.float32),
            substance_atoms] + \
            [data[('atom_neighbors', degree)] for degree in degrees] + \
            [data[('bond_neighbors', degree)] for degree in degrees]
    
        
    flat_substances_batch_list = tf.py_func(
        func=func,
        inp=[smiles],
        Tout=[
            tf.int64,                           #i=0     n_atoms
            tf.float32,                         #i=1     atom_features
            tf.float32,                         #i=2     bond_features
            tf.int64] +                         #i=3     substance_atoms
            [tf.int64 for degree in degrees] +  #i=4-9   atom_neighbors
            [tf.int64 for degree in degrees])   #i=10-15 bond_neighbors

    flat_substances_batch = {}

    n_atoms = flat_substances_batch_list[0]

    atom_features = flat_substances_batch_list[1]
    atom_features.set_shape([None, n_atom_features])
    flat_substances_batch['atom_features'] = atom_features

    bond_features = flat_substances_batch_list[2]
    bond_features.set_shape([None, n_bond_features])
    flat_substances_batch['bond_features'] = bond_features

    substance_atom_indices = flat_substances_batch_list[3]
    substance_atom_values = tf.fill(tf.expand_dims(tf.to_int32(n_atoms), 0), 1.0)
    substance_atom_shape = [eval_params['batch_size'], n_atoms]
    flat_substances_batch['substance_atoms'] = tf.SparseTensor(
        substance_atom_indices,
        substance_atom_values,
        substance_atom_shape)



    i = 4
    for degree in degrees:
        atom_neighbors = flat_substances_batch_list[i]
        atom_neighbors.set_shape([None, degree])
        flat_substances_batch['atom_neighbors_{}'.format(degree)] = atom_neighbors
        i +=1

    for degree in degrees:
        bond_neighbors = flat_substances_batch_list[i]
        bond_neighbors.set_shape([None, degree])
        flat_substances_batch['bond_neighbors_{}'.format(degree)] = bond_neighbors
        i +=1

    return flat_substances_batch


def smiles_labels_batch_queue(eval_params):
    fname_queue = tf.train.string_input_producer(
        [eval_params['substances_fname']],
        num_epochs=None,
        shuffle=True,
        name="substances_fname_queue")
    
    reader = tf.TextLineReader(
        skip_header_lines=1,
        name="substance_file_reader")
    _, record = reader.read(queue=fname_queue)
    substance_id, smiles, label = tf.decode_csv(
        records=record,
        record_defaults=[[""], [""], [1.0]],
        field_delim=eval_params['substances_field_delim'])
    smiles_batch, labels_batch = tf.train.shuffle_batch(
        tensors = [smiles, label],
        batch_size = eval_params['batch_size'],
        capacity = eval_params['queue_capacity'],
        min_after_dequeue = eval_params['queue_min_after_dequeue'],
        num_threads = eval_params['queue_num_threads'],
        seed = eval_params['queue_seed'])
    return smiles_batch, labels_batch


