#!/usr/bin/env python3.8.5
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
from setuptools import setup, find_packages
import os

# find the directory and work backwards 1 level to find readme
LOCAL_USER = os.getlogin()
README_PATH = f'C:\\Users\\{LOCAL_USER}\\source\\repos\\KMeansClusteringPython\\README.md'

setup(
    name="my_k_means",

    version='0.2.9',

    license='None',

    author='James Rose',

    author_email='JamesERose8@gmail.com',

    description=(
        'A program that can cluster vectors using k means'
        ' strategy to group likewise data.'),

    classifiers=[
        'Development status :: 0 - Pre-alpha',

        'Intended Audience :: Data Engineers, Scientists, Software Engineers',

        'Topic :: Software Development :: Libraries',

        'Programming Language :: Python - Anaconda :: 3.8.5'
    ],

    packages=find_packages(),

    long_description=open(README_PATH).read(),

    zip_safe=False,

    setup_requires=[],

    test_suite='')