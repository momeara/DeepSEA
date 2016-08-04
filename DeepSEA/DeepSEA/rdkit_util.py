#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:


# adapt neural-fingeprint rdkit_util
# to fit fingerprints based on a single endpoint regression


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import AllChem



def smiles_to_fps(smiles_tuple, fp_radius, fp_length):
	fps = []
	for smiles in smiles_tuple:
		molecule = Chem.MolFromSmiles(smiles)
		fp = AllChem.GetMorganFingerprintAsBitVect(
			molecule, fp_radius, nBits=fp_length)
		fps.append(fp.ToBitString())
	fps = np.array(fps)
	fps = np.array([list(fp) for fp in fps], dtype=np.float32)
	return fps
