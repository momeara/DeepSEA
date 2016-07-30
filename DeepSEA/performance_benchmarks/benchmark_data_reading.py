#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

import os
from time import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
import tensorflow as tf

@contextmanager
def tictoc():
    print("--- Start clock ---")
    t1 = time()
    yield
    dt = time() - t1
    print("--- Stop clock: {} seconds elapsed ---".format(dt))

def test_input_producer(fname):
    import pdb
    pdb.set_trace()

    with tf.Session() as sess:
        strings = [b"to", b"be", b"or", b"not", b"to", b"be"]
        num_epochs = 3
        queue = tf.train.string_input_producer(
            strings, num_epochs=num_epochs, shuffle=False)
        dequeue_many = queue.dequeue_many(len(strings) * num_epochs)
        dequeue = queue.dequeue()
        tf.initialize_all_variables().run()
        tf.initialize_local_variables().run()
        threads = tf.train.start_queue_runners()

        # No randomness, so just see repeated copies of the input.
        output = dequeue_many.eval()
        self.assertAllEqual(strings * num_epochs, output)

        # Reached the limit.
        with self.assertRaises(tf.errors.OutOfRangeError):
            dequeue.eval()
        for thread in threads:
            thread.join()


def test_input_fname_producer(input_fname):
    import pdb
    pdb.set_trace()

    with tf.Session() as sess:
        queue = tf.train.string_input_producer(
            [input_fname], num_epochs=None, shuffle=False)
        dequeue = queue.dequeue()
        tf.initialize_all_variables().run()
        tf.initialize_local_variables().run()
        threads = tf.train.start_queue_runners()
        output = dequeue.eval()
        for thread in threads:
            thread.join()



def read_data_int64(input_fname):
    import pdb
    with tictoc():
        input_fname_queue = tf.train.string_input_producer([input_fname], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(input_fname_queue)
        features = {'bit_features' : tf.VarLenFeature(tf.int64)}
        parsed_example = tf.parse_single_example(serialized_example, features)
        bit_features = parsed_example['bit_features']
        bit_features = tf.sparse_tensor_to_dense(bit_features)
        bit_features = tf.reshape(bit_features, [-1, 62])

        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            tf.initialize_local_variables().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                i = 0
                while not coord.should_stop():
                    x = bit_features.eval()
                    if i % 10000 == 0: print("substance {}".format(i))
                    i += 1
            except tf.errors.OutOfRangeError:
                pass
            finally:
                coord.request_stop()

            coord.join(threads)


if __name__ == '__main__':
#    test_input_producer()
#    test_input_fname_producer("/scratch/momeara/chembl_prep_int64.tfrecords")
    read_data_int64("/scratch/momeara/chembl_prep_int64_2.tfrecords")
#    read_data_sparse("/scratch/momeara/chembl_prep_sparse.tfrecords")
#    read_data_packed("/scratch/momeara/chembl_prep_packed.tfrecords")
