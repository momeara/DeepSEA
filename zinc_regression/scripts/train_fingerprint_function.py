#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

# adapt neural-fingeprint/examples/regression.py
# to fit fingerprints classify zinc catalogs

from __future__ import print_function

import pickle
import sys
import importlib
from argparse import ArgumentParser

from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
import neuralfingerprint
from neuralfingerprint import load_data
from neuralfingerprint import normalize_array, adam
from neuralfingerprint import build_batched_grad
from neuralfingerprint.build_vanilla_net import build_fingerprint_deep_net
from neuralfingerprint.build_convnet import build_convnet_fingerprint_fun

# Initialize user parameters

#model_params = dict(
#    fp_length = 512,   # Usually neural fps need far fewer dimensions than morgan.
#    fp_depth = 4,      # The depth of the network equals the fingerprint radius.
#    fp_width = 20,     # Only the neural fps need this parameter.
#    h1_size = 100,     # Size of hidden layer of network on top of fps.
#    L2_reg = np.exp(-2),
#    normalize=True,
#    nll_func_name = "rmse",
#    nll_func = rmse,
#    layer_sizes = [215, 100], # one hidden layer <- [fp_length, h1_size]
#)

##vanilla_net_params
#net_params = dict(
#    layer_sizes = [model_params['fp_length'], model_params['h1_size']],  # One hidden layer.
#    normalize=True,
#    L2_reg = model_params['L2_reg'],
#    nll_func_name = "rmse",
#    nll_func = rmse,
#)

#train_params = dict(
#    num_iters = 10,
#    batch_size = 100,
#    init_scale = np.exp(-4),
#    step_size = np.exp(-6),
#    b1 = np.exp(-3),   # Parameter for Adam optimizer.
#    b2 = np.exp(-2),   # Parameter for Adam optimizer.
#)

def load_nll_func(nll_func_name):
    mod_name, func_name = nll_func_name.rsplit('.',1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)


def print_performance(target_name, predict_func, nll_func_name, data):
    nll_func = load_nll_func(nll_func_name)
    print("\nPerformance ({}) on {}:".format(nll_func_name, target_name))
    print("\tTrain {}: {}".format(nll_func_name, nll_func(predict_func(data[0][0]), data[0][1])))
    print("\tValidation {}: {}".format(nll_func_name, nll_func(predict_func(data[1][0]),  data[1][1])))
    print("-" * 80)

# adapted from neural-fingerprints/examples/regression.py
def train_nn(
        pred_fun,
        loss_fun,
        nll_func_name, nll_func,
        num_weights,
        train_smiles,
        train_raw_targets,
        train_params,
        seed=0,
        validation_smiles=None,
        validation_raw_targets=None):
    """loss_fun has inputs (weights, smiles, targets)"""
    print("Total number of weights in the network: {}".format(num_weights))

    init_weights = npr.RandomState(seed).randn(num_weights) * train_params['init_scale']

    train_targets, undo_norm = normalize_array(train_raw_targets)

    training_curve = []
    def callback(weights, iter):
        if iter % 10 == 0:
            print("Iteration {}".format(iter))
            print("\tmax of weights: {}".format(np.max(np.abs(weights))))

            cur_loss = loss_fun(weights, train_smiles, train_targets)
            training_curve.append(cur_loss)
            print("\tloss {}".format(cur_loss))

            train_preds = undo_norm(pred_fun(weights, train_smiles))
            print("\ttrain {}: {}".format(
                nll_func_name, nll_func(train_preds, train_raw_targets)))

            if validation_smiles is not None:
                validation_preds = undo_norm(pred_fun(weights, validation_smiles))
                print("\tvalidation {}: {}".format(
                    nll_func_name, nll_func(validation_preds, validation_raw_targets)))

    # Build gradient using autograd.
    grad_fun = grad(loss_fun)
    grad_fun_with_data = build_batched_grad(grad_fun, train_params['batch_size'],
                                            train_smiles, train_targets)


    # Optimize weights.
    trained_weights = adam(grad_fun_with_data, init_weights, callback=callback,
                           num_iters=train_params['num_iters'], step_size=train_params['step_size'],
                           b1=train_params['b1'], b2=train_params['b2'])

    def predict_func(new_smiles):
        """Returns to the original units that the raw targets were in."""
        return undo_norm(pred_fun(trained_weights, new_smiles))
    return predict_func, trained_weights, training_curve


def fit_fingerprints(task_params, model_params, train_params, verbose):
    if verbose:
        print("Loading data from '{data_fname}' with\n\tsmiles column: '{smiles_column}'\n\ttarget column: '{target_column}'\n\tN_train: {N_train}\n\tN_validate: {N_validate}\n\tN_test: {N_test}\n".format(**task_params))

    data = load_data(
        filename=task_params['data_fname'],
        sizes=(task_params['N_train'], task_params['N_validate'], task_params['N_test']),
        input_name=task_params['smiles_column'],
        target_name=task_params['target_column'])

    if verbose:
        print("Building fingerprint function of length {fp_length} as a convolutional network with width {fp_width} and depth {fp_depth} ...".format(**model_params))

    # Build deep convolutional neural network that when instantiated
    # with weights, take a list of smiles produces fingerprint vectors
    # for each.
    #   weights type: WeightsParser
    #   smiles type: Iterable[str]
    #   output type: ndarray[??]    # the see output_layer_fun_and_atom_activations function
    #   fp_func type: Callable[[weights, smiles], output]
    #   fp_parser type: WeightsParser
    fp_func, fp_parser = \
        build_convnet_fingerprint_fun(
            num_hidden_features = [model_params['fp_width']] * model_params['fp_depth'],
            fp_length = model_params['fp_length'],
            normalize = True)

    if verbose:
        print("Building regression network ... ")

    # Builds a deep convolutional neural netowrk that stacks neural
    # fingerprint network on top of a vanilla convolutional network
    # with regularlized L2 loss function underneath
    #   loss_fun type: Callable[[weights, smiles, targets], numeric]
    #   pred_fun type: Callable[[weights, smiles], np.array]
    #   combined_parser: WeightsParser
    net_params = dict(
        layer_sizes = [
            model_params['fp_length'], model_params['h1_size']],
        normalize = True,
    	L2_reg = np.exp(model_params['log_l2_penalty']),
        nll_func = model_params['nll_func'])
    loss_fun, pred_fun, combined_parser = \
        build_fingerprint_deep_net(
            net_params=net_params,
            fingerprint_func=fp_func,
            fp_parser=fp_parser,
            fp_l2_penalty=np.exp(model_params['log_l2_penalty']))

    if verbose:
        print("Training model ...")
    # Train the full network for the activity using the training data
    # optimizing the loss over the validation data
    #   predict_func type: Callable[[smiles], np.array]
    #   trained_weights type: np.ndarray
    #   training_curve type: Iterable[numeric]
    predict_func, trained_weights, training_curve = \
        train_nn(
            pred_fun=pred_fun,
            loss_fun=loss_fun,
            nll_func_name=model_params['nll_func_name'],
            nll_func=model_params['nll_func'],
            num_weights=len(combined_parser),
            train_smiles=data[0][0],
            train_raw_targets=data[0][1],
            train_params=train_params,
            seed=task_params['seed'],
            validation_smiles=data[1][0],
            validation_raw_targets=data[1][1])

    if verbose:
        print_performance(
            target_name=task_params['target_column'],
            predict_func=predict_func,
            nll_func_name=model_params['nll_func_name'],
            data=data)

    trained_fp_weights = combined_parser.get(trained_weights, 'fingerprint weights')
    return trained_fp_weights, training_curve



def save_model(model_params, weights, fname):
    del(model_params['nll_func'])
    with open(fname, 'wb') as f:
         pickle.dump(dict(model_params=model_params, weights=weights), f)

def save_training_curve(training_curve, fname):
    with open(fname, 'w') as f:
        f.write("loss\n")
        [f.write("{}\n".format(loss)) for loss in training_curve]

def main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr):
    parser = ArgumentParser("Train a neural fingerprint function")

    # paths etc.
    parser.add_argument("--input_data_fname", help="Comma separated value file of substance activity data. After a header row, each row represents a substance and having columns identified by --smiles_column and --activity_column", default="data.csv")
    parser.add_argument("--output_fp_function_fname", help="Name of fingerprint function output file", default="fp_function.pickle")
    parser.add_argument("--output_training_curve_fname", help="Name of training curve output file", default="training_curve.tsv")
    parser.add_argument("--verbose", default=False, action='store_true', help="Report verbose output")

    #task_params
    parser.add_argument("--smiles_column", help="Name of substance smiles column.", default="smiles")
    parser.add_argument("--target_column", help="Name of substance target column.", default="target")
    parser.add_argument("--N_train", help="Number of substance to use for model training.", default=80, type=int)
    parser.add_argument("--N_validate", help="Number of substances to use for model validation.", default=20, type=int)
    parser.add_argument("--N_test", help="Number of substances to use for model testing.", default=20, type=int)
    parser.add_argument("--seed", help="Random seed used in training.", default=0, type=int)


    # model params
    parser.add_argument("--fp_length", help="Number of elements in the fingerprint vector", default=512, type=int)
    parser.add_argument("--fp_depth", help="Depth of fingerprint neural network", default=4, type=int)
    parser.add_argument("--fp_width", help="Width of fingerprint neural network", default=20, type=int)
    parser.add_argument("--h1_size", help="Size of hidden layer of network on top of fingerprints", default=100, type=int)
    parser.add_argument("--log_l2_penalty", help="Log Regularization factor for loss in neural network", default=-2, type=float)
    parser.add_argument("--nll_func_name", help="Name of negative log likelihood loss criterion for fingerprint neural network", default="neuralfingerprint.util.rmse")
    parser.add_argument("--layer_sizes", help="vector of layer sizes for fingerprint neural network", default=[215, 100], type=int, nargs='+')

    # train params
    parser.add_argument("--num_iters", help="Training Number of iterations", default=10, type=int)
    parser.add_argument("--batch_size", help="Training batch size", default=100, type=int)
    parser.add_argument("--log_init_scale", help="Training log initial scale", default=-4, type=float)
    parser.add_argument("--log_step_size", help="Training log step size", default=-6, type=float)
    parser.add_argument("--log_b1", help="Training log Adam optimizer parameter b1", default=-3, type=float)
    parser.add_argument("--log_b2", help="Training log Adam optimizer parameter b2", default=-2, type=float)

    params, others = parser.parse_known_args(args)

    task_params = dict(
        data_fname = params.input_data_fname,
        N_train = params.N_train,
        N_validate = params.N_validate,
        N_test = params.N_test,
    	smiles_column = params.smiles_column,
        target_column = params.target_column,
	seed = params.seed)

    model_params = dict(
        fp_length = params.fp_length,
        fp_depth = params.fp_depth,
        fp_width = params.fp_width,
        h1_size = params.h1_size,
        log_l2_penalty = params.log_l2_penalty,
        nll_func_name = params.nll_func_name,
        nll_func = load_nll_func(params.nll_func_name),
        layer_sizes = params.layer_sizes)

    train_params = dict(
        num_iters = params.num_iters,
        batch_size = params.batch_size,
        init_scale = np.exp(params.log_init_scale),
        step_size = np.exp(params.log_step_size),
        b1 = np.exp(params.log_b1),
        b2 = np.exp(params.log_b2))


    trained_fp_weights, training_curve = \
        fit_fingerprints(
            task_params=task_params,
            model_params=model_params,
            train_params=train_params,
            verbose=params.verbose)

    save_model(model_params, trained_fp_weights, params.output_fp_function_fname)
    save_training_curve(training_curve, params.output_training_curve_fname)



if __name__ == '__main__':
    sys.exit(main(args=sys.argv[1:], stdout=sys.stdout, stderr=sys.stderr))


