#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:

# adapt neural-fingeprint/examples/regression.py
# to fit fingerprints based on a single endpoint regression

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


def rmse(predictions, labels):
	return np.sqrt(np.mean((labels - predictions)**2))


