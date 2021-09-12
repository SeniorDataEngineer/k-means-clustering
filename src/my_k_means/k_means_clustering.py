#!/usr/bin/env python3.8.5
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
import random
from decimal import *

# Third party imports.
import numpy


class KMeans():
    """
    This class can be used to instantiate a mini-toolkit for running k means
    clustering analysis of data sets.
    """

    def __init__(
            self,
            start_k: int = 1,
            seek_ideal_k: bool = False,
            k_fitting_method: str = 'silh',
            decimal_precision: int = 3):
        """
        Initializes a KMeans clustering object. Call get_clusters to trigger
        algorithm. \n
        Keyword arguments:
            start_k          -- a starting value for k
            seek_ideal_k     -- whether to search for ideal k
            k_fitting_method -- analytical method to qualitatively
                                decide on k.
        """
        self.start_k = start_k
        self.seek_ideal_k = seek_ideal_k
        self.k_test_method = k_fitting_method
        self.cluster_centroids_labels = []
        self.silhouette_scores = []
        getcontext().prec = decimal_precision

    def get_euclidean_distance(
            self,
            p: list,
            q: list) -> [float]:
        """
        Given 2 n-dimensional coordinates calculate distance for each ith list
        using pythagorean theorem. Return the distance as a float. \n
        Returns:
            [float]
        Doctest:
            >>> km = KMeans()
            >>> p, q, e = [1,1], [3,3], 2.83
            >>> assert round(km.get_euclidean_distance(p, q), 2) == e
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
            >>> r = km.pair_closest_vectors(p, c, km.get_euclidean_distance)
            >>> assert numpy.all(r == e)
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
            retry_limit: int = 5) -> [int]:
        """
        For an input of len_ length pick random element positions and return
        those positions. The method will retry a call to random 5 tries before
        giving up and returning a duplicate. \n
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
        Given a list of n-dimensional floats return the mean for each
        dimension. \n
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
            for v in vectors:
                mean[i] += v[i]
            mean[i] /= len(vectors)
        return mean

    def get_clusters(
            self,
            k: int,
            vectors: [[float]],
            max_iterations: int = 100) -> ([[float]], [[float]]):
        """
        Given a series of vectors return a list of labels and a list of center
        vectors. \n
        Returns:
            ([[float]], [[float]])
        Usage:
            centroids, labels = get_clusters(...)
        """
        # Pick k centroids randomly.
        centroids = [
            vectors[i]
            for i in self.get_items_randomly(len(vectors)-1, k)]

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
        Splits a numpy.ndarray at index i excluding ith element of the array.
        \n
        Returns:
            numpy.ndarray
        Doctest:
            >>> km = KMeans()
            >>> a = numpy.array([1,2,3,4])
            >>> assert numpy.all(km.ndarray_without_ith(a, 0) == [2, 3, 4])
            >>> assert numpy.all(km.ndarray_without_ith(a, 1) == [1, 3, 4])
            >>> assert numpy.all(km.ndarray_without_ith(a, 3) == [1, 2, 3])
        """
        arr_1 = array[0: i]
        arr_2 = array[i+1:]
        if len(arr_1) == 0:
            return arr_2
        elif len(arr_2) == 0:
            return arr_1
        return numpy.concatenate(
                    (arr_1, arr_2),
                    axis=0)

    def get_arithmetic_mean(
            self,
            series: [float]) -> float:
        """
        Given a series of int return the mean/average for that series.
        If import_ is supplied a library is used for the calculation. \n
        Returns:
            float
        DocTest:
            >>> km = KMeans()
            >>> assert km.get_arithmetic_mean([1,2,3]) == 2.0
        """
        return float(
                     sum(series) / len(series))

    def get_silhouette_coefficient(
            self,
            vectors: [[float]],
            labels: [int],
            centroids: [[float]]) -> ([float], [float]):
        """
        Given a list of vectors, labels for those vectors and the
        corresponding list of centroids, return the silhouette
        coefficients indicating whether the selected k is a good fit. \n
        Returns:
            [float]
        Doctest:
            >>> km = KMeans()
            >>> v = numpy.array([[1,1], [2,2], [8,8], [9,9]])
            >>> l = numpy.array([0,0,1,1])
            >>> c = numpy.array([[1.5,1.5], [8.5,8.5]])
            >>> r = km.get_silhouette_coefficient(v, l, c)
            >>> s = round(sum([ s for s in r[0][0] ]) / len(r[0][0]), 2)
            >>> assert s == 0.86
            >>> assert r[1] == [1, 0]
        """
        f = self.get_euclidean_distance
        g = self.get_arithmetic_mean
        pairs = []
        silhouettes = [[] for _ in range(0, len(centroids))]

        # Pair each centroid to closest other centroid.
        for i, c in enumerate(centroids):
            others = self.ndarray_without_ith(centroids, i)
            p = self.pair_closest_vectors([c], others, f)
            # Others is missing ith item.
            if not p < i:
                p += 1
            pairs.extend(p)

        for i, v in enumerate(vectors):
            j = labels[i]
            # a: intra-cluster mean distance.
            a = g([
                    f(x, v)
                    for x in vectors[labels == j]
                    if not numpy.all(x == v)])

            # b: inter-cluster distance.
            b = g([
                    f(y, v)
                    for y in vectors[labels == pairs[j]]])

            # Calculate silhouette.
            silhouettes[j].append(
                                    (b - a) / max(a, b))

        return (silhouettes, pairs)

    def get_silhouette_summary(
            self,
            i: int) -> float:
        """
        TODO : Comment, seek usage.
        Returns:
        Doctest:
        """
        f = self.get_arithmetic_mean
        s = f([
                x
                for y in self.silhouette_scores[i][0]
                for x in y])
        return s

    def get_best_fit(
            self,
            vectors: [[float]],
            start_k: int = 2,
            max_iterations: int = 100) -> (int, [[float]], [[float]]):
        """
        Iterates over possible k from 2 to 10, measures the fit of k and
        returns the best k, centroids and labels. \n
        Returns:
            (int, [[float]], [[float]])
        """
        for k in range(start_k, 11):
            # Cluster.
            cnt, lbl = self.get_clusters(k, vectors, max_iterations)
            self.cluster_centroids_labels.append((cnt, lbl))

            # Test quality of cluster fit.
            s, p = self.get_silhouette_coefficient(vectors, lbl, cnt)
            self.silhouette_scores.append((s, p))

        # Pick k.
        k = self.get_best_fit_index()

        return (
                k + 2,
                self.cluster_centroids_labels[k][0],
                self.cluster_centroids_labels[k][1])

    def get_best_fit_index(
            self,
            skip_negatives: bool = False) -> int:
        """
        Finds and returns the index of the best fitting k. Best fit is defined
        as the silhouette that has no negative coefficients and has the
        highest average silhouette
        per cluster. \n
        Returns:
            int
        Doctest:
            >>> km = KMeans()
            >>> a = ([[0.7, 0.6], [0.7, 0.6]], [1, 1, 0])
            >>> b = ([[0.7, 0.6, 0.1], [0.7, 0.6]], [1, 2, 0])
            >>> x = [a, b]
            >>> km.silhouette_scores = x
            >>> assert km.get_best_fit_index() == 0
        """
        f = self.get_arithmetic_mean
        h_mean = float('-inf')
        index = 0

        for i, s in enumerate(self.silhouette_scores):
            mean = 0
            mean = f([x for y in s[0] for x in y])
            if mean > h_mean:
                h_mean = mean
                index = i

        return index


if __name__ == "__main__":
    import doctest
    doctest.testmod()
