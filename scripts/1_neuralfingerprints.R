# -*- tab-width:2;indent-tabs-mode:t;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:


system("
cd /scratch/momeara

# Install Anaconda2.7 because neural fingerprints is not python3 compatable
# https://anaconda.org/
wget http://repo.continuum.io/archive/
bash Anaconda2-4.0.0-Linux-x86_64.sh
# enter /nfs/work/momeara/tools/anaconda2 for install path and let installer finish
# prepend /nfs/work/momeara/tools/anaconda2/bin to shell PATH variable

# https://anaconda.org/rdkit/rdkit
conda install -c rdkit rdkit=2016.03.1

pip install --upgrade pip
pip install autograd

cd ~/work/sea/DeepSEA
git clone git@github.com:HIPS/neural-fingerprint.git
cd neural-fingerprint
python setup.py install


")
