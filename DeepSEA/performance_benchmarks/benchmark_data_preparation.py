#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

import sys
from time import time
from contextlib import contextmanager
import pandas as pd
from neuralfingerprint.mol_graph import graph_from_smiles

###############
# filter valid substances for neuralfingerprint.MolGraph loading
#
# usage:
#
#   ./filter_valid_substances.py <input_subtances>.tsv <output_substances>.tsv
#
#

@contextmanager
def tictoc():
    print("--- Start clock ---")
    t1 = time()
    yield
    dt = time() - t1
    print("--- Stop clock: {} seconds elapsed ---".format(dt))


def filter_valid(input_fname, output_fname):
    print("filter valid inputs:")

	def func(row):
		substance_id, smiles, label = row.ix
		try:
			graph_from_smiles(smiles)
		except Exception as e:
			print("failed to parse compound: {} with smiles {}. error: {}".format(substance_id, smiles, e.message))
			return False
		return True

    with tictoc():
        df = pd.read_csv(input_fname, sep='\t')
        df[df.apply(func, axis=1)]
		df.write_csv(output_fname, sep='\t')


if __name__ == '__main__':

	input_fname = sys.argv[1]
	output_fname = sys.argv[2]
    filter_vaid(input_fname, output_fname)
