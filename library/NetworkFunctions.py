import numpy
import tensorflow as tf
from matplotlib import pyplot


def get_weight_mask(w=None, inverse=False, plot=False):
    if w is None:
        n = 10
    else:
        s = w.shape.as_list()
        n = s[0]
    m = numpy.ones((n, n))
    for r in range(n):
        for c in range(n):
            if c > (n / 2) - 1 and r < (n / 2): m[r, c] = 0
            if c < (n / 2) and r > (n / 2) - 1: m[r, c] = 0
    if inverse: m = numpy.abs(m - 1)
    indices = numpy.nonzero(m)
    rows = indices[0]
    cols = indices[1]
    output_list = []
    if plot: pyplot.figure()
    for r, c in zip(rows, cols):
        output_list.append([r, c])
        if plot: pyplot.scatter(r, c)
    if plot: pyplot.show()
    return output_list


class WeightConstraint(tf.keras.constraints.Constraint):
    def __init__(self, apply=True):
        self.apply = apply

    def __call__(self, w):
        indices = get_weight_mask(w, inverse=False, plot=False)
        locations = tf.constant(indices)
        updates = tf.constant([1.0] * len(indices))
        mask = tf.scatter_nd(locations, updates, w.shape)
        if self.apply: w = w * mask
        return w
