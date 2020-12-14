#!/usr/bin/env python3.9
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
from collections import namedtuple
import random

# Third party imports.
import numpy
from sklearn.metrics import pairwise_distances_argmin


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


class KMeans():
    """
    This class can be used ot instiate a mini-toolkit
    for running k means clustering analysis of data sets.
    """

    def __init__(self):
        pass

    def get_euclidean_distance(
            self,
            p: tuple,
            q: tuple) -> float:
        """
        Given 2 n-dimensional coordinates calculate distance
        for each ith tuple using pythagorean theorem. Return
        the distance as a float. \n
        Returns:
            list
        Doctest:
            >>> km = KMeans()
            >>> assert round(km.get_euclidean_distance(p=(1,1), q=(3,3)), 2) == 2.83
        """
        if not len(p) == len(q):
            raise ValueError('Both p and q must have same dimensionality.')

        return sum([
            (p[dim] - q[dim]) ** 2
            for dim in range(len(p))
        ]) ** 0.5

    def get_items_randomly(
            self,
            len_: int,
            k_items: int) -> [int]:
        """
        For an input of len_ length pick random element
        positions and return those positions. The method
        will retry a call to random 3 tries before giving
        up and returning a duplicate.
        Returs:
            [int]
        Doctest:
            >>> km = KMeans()
            >>> assert len(km.get_items_randomly(len_=300, k_items=3)) == 3
        """
        if len_ < k_items:
            return ValueError('Range cannot be smaller than required'
            ' number of possible values')
        if len_ == k_items:
            return [
                v
                for v in range(0, len(len_))
            ]

        random_positions = []
        for _ in range(0, k_items):
            rnd = random.randint(0, len_)
            i = 0
            while rnd in random_positions:
                rnd = random.randint(0, len_)
                i += 1
                if i == 3:
                    break
            random_positions.append(rnd)
        return random_positions

    def k_means_2d(
            k: int,
            points: [namedtuple],
            random_seed: int=2) -> [namedtuple]:
        """
        Clusters 2-dimensional array of points into groups and
        returns the points with their cluster assignment. \n
        Returns:
            [namedtuple]
        Doctest:
            Test this with pytest, the testcase is too complex
            to represent in doctest.
        """
        range_ = numpy.random.RandomState(random_seed)
        i = range_.permutation(points[0])
        centers = points[i]
        while True:
            labels = pairwise_distances_min(points)

    def x_k_means_2d(
            self,
            k: int,
            points: [namedtuple],
            random_seed: int=2,
            max_iterations: int=100) -> [namedtuple]:
        
        centroids = self.get_items_randomly(
            len_=len(points),
            k_items=k)

        iteration = 0
        while True:


            if iteration == max_iterations:
                break
        

        print(centroids)



if __name__ == "__main__":
    import doctest
    doctest.testmod()


    km = KMeans()
    km.x_k_means_2d(2, [(1,2),(3,3),(4,4),(4,1),(6,5)])