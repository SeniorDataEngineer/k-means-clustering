#!/usr/bin/env python3.8.5
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
import random

# Third party imports.
import numpy


class KMeans():
    """
    This class can be used to instantiate a mini-toolkit
    for running k means clustering analysis of data sets.
    """

    def __init__(
            self,
            start_k: int=3,
            seek_ideal_k: bool=False,
            k_fitting_method: str='silh'):
        """
        Initializes a KMeans clustering object. Call
        get_cluster to trigger algorithm. \n
        Keyword arguments:
            start_k          -- a starting value for k
            seek_ideal_k     -- whether to search for ideal k
            k_fitting_method -- analytical method to qualitatively
                                decide on k.
        """
        self.start_k = start_k
        self.seek_ideal_k = seek_ideal_k
        self.k_test_method = k_fitting_method
        self.fitting_silhouette_scores = []

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
            >>> assert round(km.get_euclidean_distance(p=[1,1], q=[3,3]), 2)
            >>> ... == 2.83
        """
        if not len(p) == len(q):
            raise ValueError('Both p and q must have same dimensionality.')

        return sum([
            (p[dim] - q[dim]) ** 2
            for dim in range(len(p))
        ]) ** 0.5

    def pair_closest_vectors(
            self,
            vectors_a: [[float]],
            vectors_b: [[float]],
            dist_measure: object) -> [int]:
        """
        Iterate through each vector_a and compare to each vector_b.
        If vector_b is closer by distance than all previous comparisons
        then jth index. \n
        Returns:
            [int]
        Doctest:
            >>> km = KMeans()
            >>> p = [[2, 2], [6, 6]]
            >>> c = [[1, 1], [7, 7]]
            >>> e = [0, 1]
            >>> assert km.pair_closest_vectors(p, c) == e
        """
        pos_inf = float('inf')
        closest_vb = []
        for i, va in enumerate(vectors_a):
            shortest = pos_inf
            closest_vb.append(0)
            for j, vb in enumerate(vectors_b):
                distance = dist_measure(va, vb)
                if distance < shortest:
                    shortest = distance
                    closest_vb[i] = j
        return numpy.array(closest_vb)

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
            vectors: [[float]]) -> [float]:
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
        for i in range(len(vectors[0])):
            mean.append(0)
            for p in vectors:
                mean[i] += p[i]
            mean[i] /= len(vectors)
        return mean

    def get_cluster(
            self,
            k: int,
            vectors: [[float]],
            random_seed: int=2,
            max_iterations: int=100) -> ([[float]], [[float]]):
        """
        Given a series of vectors return a list of labels and a
        list of center vectors. \n
        Returns:
            ([[float]], [[float]])
        Usage:
            centroids, labels = get_cluster(...)
        """
        # Pick k centroids randomly.
        centroids = [
            vectors[i]
            for i in self.get_items_randomly(len(vectors), k)]

        iteration = 0
        while True:
            if iteration == max_iterations:
                break
            iteration += 1

            # Get the closest centroid for each vector.
            labels = self.pair_closest_vectors(vectors, centroids,
                                               self.get_euclidean_distance)

            # Replace centroid with cluster mean.
            new_centroids = numpy.array([
                                self.get_nd_series_mean(vectors[labels == i])
                                for i in range(k)])

            # Check for convergence.
            if numpy.all(centroids == new_centroids):
                break
            centroids = new_centroids

        return (centroids, labels)

    def ndarray_without_ith(
            self,
            array: numpy.ndarray,
            i: int) -> numpy.ndarray:
        """
        Splits a numpy.ndarray at index i excluding ith element of the
        array. \n
        Returns:
            numpy.ndarray
        """
        return numpy.concatenate(
                    (array[0: i], array[i+1:]),
                    axis=0)

    def get_silhouette_coefficient(
            self,
            vectors: [[float]],
            labels: [int],
            centroids: [[float]]) -> ([float], [float]):
        """
        Given a list of vectors, labels for those vectors
        and the corresponding list of centroids, return the
        silhouette coeficients indicating whether the selected
        k is a good fit. \n
        Returns:
            [float]
        Doctest:
            >>> assert 1 == 2
        """
        # Pair the centroids to their closest partner.
        pairs = []
        for i, c in enumerate(centroids):
            others = self.ndarray_without_ith(centroids, i)
            p = self.pair_closest_vectors([c], others,
                                          self.get_euclidean_distance)
            # Others is missing ith item.
            if p < i:
                pairs.extend(p)
            else:
                pairs.extend(p+1)

        # Calculate the silhouette_coefficient for each vector.
        silhouettes = []
        for j in range(0, len(centroids)):
            silhouettes.append([])
            for v in vectors[labels == j]:
                # a: get intra-cluster mean distance.
                a = self.get_nd_series_mean([
                        [self.get_euclidean_distance(x, v)]
                        for x in vectors[labels == j]
                        if not numpy.all(x == v)])[0]

                # b: get the inter-cluster distance.
                b = self.get_nd_series_mean([
                    [self.get_euclidean_distance(x, v)]
                    for x in vectors[labels == pairs[j]]
                    if not numpy.all(x == v)])[0]

                # Calculate silhouette.
                silhouettes[j].append(
                    (b - a) / max(a, b))

        return (silhouettes, pairs)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
