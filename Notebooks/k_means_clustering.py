#!/usr/bin/env python3.9
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
import random

# Third party imports.
import numpy

class KMeans():
    """
    This class can be used ot instiate a mini-toolkit
    for running k means clustering analysis of data sets.
    """

    def __init__(self):
        pass

    def get_euclidean_distance(
            self,
            p: list,
            q: list) -> [float]:
        """
        Given 2 n-dimensional coordinates calculate distance
        for each ith list using pythagorean theorem. Return
        the distance as a float. \n
        Returns:
            [float]
        Doctest:
            >>> km = KMeans()
            >>> assert round(km.get_euclidean_distance(p=[1,1], q=[3,3]), 2) == 2.83
        """
        if not len(p) == len(q):
            raise ValueError('Both p and q must have same dimensionality.')
            
        return sum([
            (p[dim] - q[dim]) ** 2
            for dim in range(len(p))
        ]) ** 0.5

    def pair_closest_points(
            self,
            points: [float],
            centers: [[float]]) -> [int]:
        """
        Iterate through each point and compare to all centers,
        if the center is nearer than all previous centers store
        the center number. \n
        Returns:
            [int]
        Doctest:
            >>> km = KMeans()
            >>> p = [[2, 2], [6, 6]]
            >>> c = [[1, 1], [7, 7]]
            >>> e = [0, 1]
            >>> assert km.pair_closest_points(p, c) == e
        """
        pos_inf = float('inf')
        centers_by_position = []
        for i, p in enumerate(points):
            shortest = pos_inf
            centers_by_position.append(0)
            for j, c in enumerate(centers):
                distance = self.get_euclidean_distance(p, c)
                if distance < shortest:
                    shortest = distance
                    centers_by_position[i] = j
        return numpy.array(centers_by_position)

    def get_items_randomly(
            self,
            len_: int,
            k_items: int,
            retry_limit: int=5) -> [int]:
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
            >>> assert len(km.get_items_randomly(len_=3, k_items=3)) == 3
        """
        if len_ < k_items:
            return ValueError('Range cannot be smaller than required'
            ' number of possible values')
        if len_ == k_items:
            return [
                v
                for v in range(0, len_)]

        random_positions = []
        for _ in range(0, k_items):
            rnd = random.randint(0, len_)
            i = 0
            while rnd in random_positions:
                rnd = random.randint(0, len_)
                i += 1
                if i == retry_limit:
                    break
            random_positions.append(rnd)

        return random_positions

    def get_nd_series_mean(
            self,
            points: [[float]]) -> [float]:
        """
        Given a list of n-dimensional floats return the mean
        for each dimension. \n
        Returns:
            [float]
        Doctest:
            >>> km = KMeans()
            >>> s = [[2, 3], [5, 6], [9, 10]]
            >>> e = [5.333333333333333, 6.333333333333333]
            >>> assert km.get_nd_series_mean(s) == e
        """
        mean = []
        for i in range(len(points[0])):
            mean.append(0)
            for p in points:
                mean[i] += p[i]
            mean[i] /= len(points)
        return mean

    def k_means_2d(
            self,
            k: int,
            points: [[float]],
            random_seed: int=2,
            max_iterations: int=100) -> ([[float]], [[float]]):
        """
        Given a series of points return labels for each point
        and the center co-ordinates. This is a standard
        implementation of k means. \n
        Returns:
            ([[float]], [[float]])
        """
        # Pick k centroids randomly.
        centroids = [ 
            points[i] 
            for i in self.get_items_randomly(len(points), k)]

        iteration = 0
        while True:
            if iteration == max_iterations:
                break
            iteration += 1
            
            # Get the closest centroid for each point.
            labels = self.pair_closest_points(points, centroids)
            
            # Replace centroid with cluster mean.
            new_centroids = numpy.array([
                                self.get_nd_series_mean(points[labels == i])
                                for i in range(k)])

            # Check for convergence or max iterations.
            if numpy.all(centroids == new_centroids):
                break
            centroids = new_centroids

        return (centroids, labels)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
