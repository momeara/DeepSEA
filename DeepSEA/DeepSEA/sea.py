#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:


import tensorflow as tf





def main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr):
	parser = ArgumentParser("Train a neural fingerprint function")

	parser.add_argument("--input_substance_fname", help="Comma separated value file of substance data. After a header row, each row represents a substance and having columns identified by --smiles_column", default="data.csv")

	parser.add_argument("--input_sets_fname", help="Comma separated value file of substance data. After a header row, each row represents a substance and having columns identified by --smiles_column", default="data.csv")



if __name__ == '__main__':
	sys.exit(main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr))


















