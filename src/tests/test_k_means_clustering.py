#!/usr/bin/env python3.9
# Copyright 2020, Rose Software Ltd, All rights reserved.

# Built-in imports.
import os
from collections import namedtuple

# Project import.
from my_k_means.k_means_clustering import k_means_2d

# Third party imports.


# Setup for testing.
LOCAL_USER = os.getlogin()


class TestKMMeans2D:
    """
    Test the function k_means_2d in k_means_clustering.
    The results are non-deterministic so a qualitative
    approach is required.
    """

    def test_k_means_2d(self):
        """
        """
        k = 2
        Point = namedtuple('Point', 'x y cl')
        test_case_points = [
            Point(1, 2, None),
            Point(1, 3, None),
            Point(2, 2, None),
            Point(5, 6, None),
            Point(7, 4, None),
            Point(6, 4, None),
            Point(5, 6, None)
        ]

