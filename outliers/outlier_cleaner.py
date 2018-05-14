#!/usr/bin/python

import math as math

import numpy as np


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    # your code goes here
    errors = predictions - net_worths
    errors = errors.__abs__()
    input_data = np.concatenate((ages, net_worths, errors), axis=1)
    print(len(input_data))
    input_data = input_data[input_data[:, 2].argsort()]
    cleaned_data = input_data[:81, ]
    cleaned_data = tuple(map(tuple, cleaned_data))
    return cleaned_data
