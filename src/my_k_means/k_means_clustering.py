#!/usr/bin/env python3.9
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
from collections import namedtuple

# Third party imports.
#import numpy


@DeprecationWarning
class Point():
    """
    Class used to define points in space, the dimensionality
    can be defined in the initialiser.
    """

    def __init__(
        self,
        coordinates: namedtuple):
        """
        Initializes a point with coordinates.
        """
        self.dimensions = len(coordinates)


def k_means_2d(
    k: int,
    points: [namedtuple],
    random_seed: int=2) -> [tuple]:
        """
        Clusters 2-dimensional array of points into groups
        and returns the points with their cluster assignment. \n
        Returns:
            [tuple]
        Doctest:
            Test this with pytest, the testcase is too complex
            to represent in doctest.
        """
        range_ = numpy.random.RandomState(random_seed)
        print(range_)

k_means_2d(2, [(1)], 2)