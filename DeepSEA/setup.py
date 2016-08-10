#!/usr/bin/env python
# -*- tab-width:4;indent-tabs-mode:f;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=4 et sw=4:


from setuptools import setup

setup(
    name='DeepSEA',
    description='Similarity Ensemble Approach with deep learning substance fingerprints',
    url='http://github.com/momeara/DeepSEA',
    author="Matthew O'Meara",
    author_email="mattjomeara@gmail.com",
    license="MIT",
    packages=['DeepSEA'],
	scripts = [
		'bin/fit_fingerprints',
		'bin/fit_activity_triples',
		'bin/filter_valid_substances'],
    install_requires=[
        'numpy',
        'tensorflow',
        'neuralfingerprint'],
	test_suite = 'tests')
