#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




def main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr):
	parser = ArgumentParser("Train a neural fingerprint function")

	# paths etc.
	parser.add_argument("--input_data_fname", help="Comma separated value file of substance activity data. After a header row, each row represents a substance and having columns identified by --smiles_column and --activity_column", default="data.csv")
#	parser.add_argument("--output_fp_function_fname", help="Name of fingerprint function output file", default="fp_function.pickle")
	parser.add_argument("--summaries_dir", help="Name of directory where summary data should be deposited", default="logs")
	parser.add_argument("--output_training_curve_fname", help="Name of training curve output file", default="training_curve.tsv")
	parser.add_argument("--verbose", default=False, action='store_true', help="Report verbose output")

	#task_params
	parser.add_argument("--smiles_column", help="Name of substance smiles column.", default="smiles")
	parser.add_argument("--target_column", help="Name of substance target column.", default="target")
	parser.add_argument("--N_train", help="Number of substance to use for model training.", default=80, type=int)
	parser.add_argument("--N_validate", help="Number of substances to use for model validation.", default=20, type=int)
	parser.add_argument("--N_test", help="Number of substances to use for model testing.", default=20, type=int)
	parser.add_argument("--device", help="Specify the device to use.", default='cpu:0')
#	parser.add_argument("--seed", help="Random seed used in training.", default=0, type=int)


	# model params
	parser.add_argument("--fp_length", help="Number of elements in the fingerprint vector", default=512, type=int)
	parser.add_argument("--fp_depth", help="Depth of fingerprint neural network", default=4, type=int)
	parser.add_argument("--fp_width", help="Width of fingerprint neural network", default=20, type=int)
	parser.add_argument("--h1_size", help="Size of hidden layer of network on top of fingerprints", default=100, type=int)
	parser.add_argument("--l2_penalty", help="scale of l2 regularization factor for loss in neural network", default=.01, type=float)
	parser.add_argument("--l1_penalty", help="scale of l1 regularization factor for loss in neural network", default=0.0, type=float)
	parser.add_argument("--prediction_layer_sizes", help="vector of layer sizes for fingerprint neural network", default=[512, 100], type=int, nargs='+')

	# train params
	parser.add_argument("--epochs", help="Number of training epochs", default = 10, type=int)
	parser.add_argument("--batch_size", help="Training data batch size", default=100, type=int)
	parser.add_argument("--eval_batch_size", help="Batch size when evluating performance", default=64, type=int)
	parser.add_argument("--eval_frequency", help="How often to evaluate performance", default=10, type=int)
#	parser.add_argument("--log_init_scale", help="Training log initial scale", default=-4, type=float)
	parser.add_argument("--log_learning_rate", help="Training log learning rate", default=-6, type=float)
	parser.add_argument("--log_b1", help="Training log Adam optimizer parameter b1", default=-3, type=float)
	parser.add_argument("--log_b2", help="Training log Adam optimizer parameter b2", default=-2, type=float)

	params, others = parser.parse_known_args(args)

	task_params = dict(
		data_fname = params.input_data_fname,
		summaries_dir = params.summaries_dir,
		N_train = params.N_train,
		N_validate = params.N_validate,
		N_test = params.N_test,
		smiles_column = params.smiles_column,
		target_column = params.target_column,
		device = params.device)
#		seed = params.seed)

	model_params = dict(
		fp_length = params.fp_length,
		fp_depth = params.fp_depth,
		fp_width = params.fp_width,
		h1_size = params.h1_size,
		l2_penalty = params.l2_penalty,
		l1_penalty = params.l1_penalty,
		prediction_layer_sizes = params.prediction_layer_sizes)
#		nll_func_name = params.nll_func_name,
#		nll_func = load_nll_func(params.nll_func_name),


	print("GIT REVISION: {}".format(subprocess.check_output(['git', 'rev-parse', 'HEAD'])))

	train_params = dict(
		epochs = params.epochs,
		batch_size = params.batch_size,
		eval_batch_size = params.eval_batch_size,
		eval_frequency = params.eval_frequency,
#		log_init_scale = params.log_init_scale,
		log_learning_rate = params.log_learning_rate,
		log_b1 = params.log_b1,
		log_b2 = params.log_b2)


	curves = \
		fit_fingerprints(
			task_params=task_params,
			model_params=model_params,
			train_params=train_params,
			verbose=params.verbose)

	with open(params.output_training_curve_fname, 'w') as f:
		f.write("train_loss,train_rmse,validation_rmse\n")
		[f.write("{},{},{}\n".format(train_loss, train_rmse, validation_rmse)) for train_loss,train_rmse, validation_rmse in zip(curves[0], curves[1], curves[2])]


if __name__ == '__main__':
	sys.exit(main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr))
