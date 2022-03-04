import keras
import numpy
import tensorflow as tf


class weight_constraint(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self):
        pass

    def __call__(self, w):
        shape = w.shape.as_list()
        row_indices = [1, 2, 3]
        col_indices = [10, 11, 12]
        indices = numpy.ravel_multi_index((row_indices, col_indices), dims=shape, order='C')
        # Question: how to set the values indicated by "indices" to zero?
        return w


full_model = keras.Sequential()
input_layer = keras.layers.Dense(units=26, input_shape=(26,),
                                 activation='tanh',
                                 kernel_initializer='zeros',
                                 bias_initializer='zeros',
                                 kernel_constraint=weight_constraint())

full_model.add(input_layer)
loss = keras.losses.MeanSquaredError()
full_model.compile('RMSprop', loss=loss)

inputs = numpy.random.random((100, 26))
targets = numpy.random.random((100, 26))

training_history = full_model.fit(inputs, targets, epochs=10)
