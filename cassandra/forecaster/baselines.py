from random import random

import numpy as np


def gaussian_noise(df, xnew):
    n = len(xnew)
    y = df.price.values
    return (y.mean() + y.std() * 10 * np.random.rand(n)).tolist()


def random_walk(df, xnew):
    n = len(xnew)
    y = df.price.values
    initial_value = y[-1]
    delta = y.std()
    result = [initial_value]
    for i in range(1, n):
        movement = -delta if random() < 0.5 else delta
        value = result[i - 1] + movement
        result.append(value)
    return result[1:]


def naive_forecast(df, xnew):
    n = len(xnew)
    y = df.price.values
    return [*y[-n:][::-1]]
