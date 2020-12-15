#!/usr/bin/env python3.9
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
from collections import namedtuple
import random

# Third party imports.
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

    def pair_closest_points(
            self,
            points: [dict],
            centers: [dict],
            center_label: str) -> [dict]:
        """
        Given a series of points update the label to the
        center point that is closest to the point. Iterate
        through each point and compare to all centers, if
        the center is nearer than all previous centers store
        the center number in the point dict. \n
        Returns:
            [dict]
        Doctest:
            >>> km = KMeans()
            >>> p = [{'x': 2, 'y': 2, 'c': None}, {'x': 6, 'y': 6, 'c': None}]
            >>> c = [{'x': 1, 'y': 1, 'c': 1}, {'x': 7, 'y': 7, 'c': 2}]
            >>> e = [{'x': 2, 'y': 2, 'c': 1}, {'x': 6, 'y': 6, 'c': 2}]
            >>> assert km.pair_closest_points(p, c, 'c') == e
        """
        pos_inf = float('inf')
        for point in points:
            shortest_distance = pos_inf
            for center in centers:
                p, q = (point['x'], point['y']), (center['x'], center['y'])
                distance = self.get_euclidean_distance(p, q)
                if distance < shortest_distance:
                    shortest_distance = distance
                    point[center_label] = center[center_label]
        return points

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
                for v in range(0, len(len_))]

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

    def get_nd_series_mean(
            self,
            series: [dict],
            keys: [str]) -> dict:
        """
        Given a n-dimensional series of dicts return
        the mean for each dimension. \n
        Returns:
            dict
        Doctest:
            >>> km = KMeans()
            >>> s = [{'x':2, 'y':3}, {'x':5, 'y':6}, {'x':9, 'y':10}]
            >>> e = {'x':5.333333333333333, 'y':6.333333333333333}
            >>> assert km.get_nd_series_mean(series=s, keys=['x', 'y']) == e
        """
        mean = {}
        for key in keys:
            mean[key] = 0
            for item in series:
                mean[key] += item[key]
            mean[key] /= len(series)
        return mean

    def k_means_2d(
            self,
            k: int,
            points: [dict],
            label_key: str,
            keys: [str],
            random_seed: int=2,
            max_iterations: int=100) -> [dict]:
        """
        Given a series of points return k clusters of grouped
        points. This is a standard implementation of k means. \n
        Returns:
            [dict]
        """
        # Pick k centroids randomly.
        centroids = [ 
            points[i] 
            for i in self.get_items_randomly(len(points),k)]

        iteration = 0
        while True:
            if iteration == max_iterations:
                break
            iteration += 1

            # Get the closest centroid for each point.
            points = self.pair_closest_points(
                points=points,
                centers=centroids,
                center_label='c')
            print(points)

            # Replace centroid with cluster mean.
            new_centroids = []
            for c in centroids:
                grp_centroid = []
                for p in points:
                    if p.get(label_key) == c.get(label_key):
                        grp_centroid.append(p)
                new = self.get_nd_series_mean(series=grp_centroid, keys=keys)
                new[label_key] = c.get(label_key)
                new_centroids.append(new)
            print(new_centroids)

            # Check for convergence or max iterations.
            if centroids == new_centroids:
                break
            centroids = new_centroids

        return points


if __name__ == "__main__":
    import doctest
    doctest.testmod()
