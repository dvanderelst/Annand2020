import keras
import numpy
from matplotlib import pyplot

import Misc
import NetworkFunctions

pyplot.rcParams['text.usetex'] = True

color1 = '#fc8d59'
color2 = '#91bfdb'
apply_constraints = False
training_examples = 500
loss = keras.losses.MeanSquaredError()

# %% Train Network

network_model = keras.Sequential()
constraint = NetworkFunctions.weight_constraint(apply_constraints)
input_layer = keras.layers.Dense(units=26, input_shape=(26,), activation='tanh', kernel_constraint=constraint)
network_model.add(input_layer)
network_model.compile('adam', loss=loss)

inputs, outputs = Misc.create_training_examples(frequency=3, n=training_examples)
training_history = network_model.fit(inputs, outputs, epochs=500)

# %% Visualise output

selected_example = 250
pyplot.figure(figsize=(9, 3))

for test_condition in [0, 1, 2]:
    test_inputs = inputs * 1.0

    size = test_inputs[:, 0:13]
    size = size.shape
    noise = ((numpy.random.random(size) * 2) - 1)
    noise = noise

    if test_condition == 1: test_inputs[:, 0:13] = noise
    if test_condition == 2: test_inputs[:, 13:] = noise

    prediction = network_model.predict(test_inputs)

    inputs0 = test_inputs[:, 0:13]
    inputs1 = test_inputs[:, 13:]
    predictions0 = prediction[:, 0:13]
    predictions1 = prediction[:, 13:]

    output0 = outputs[:, 0:13]
    output1 = outputs[:, 13:]

    input_x = numpy.arange(13) * 0.1
    prediction_x = (numpy.arange(13) + 13) * 0.1

    pyplot.subplot(1, 4, test_condition + 1)
    pyplot.plot(input_x, inputs0[selected_example, :], '.-', color=color1)
    pyplot.plot(prediction_x, output0[selected_example, :], color='k', linewidth=3, alpha=0.5)
    pyplot.plot(prediction_x, predictions0[selected_example, :], '--', color=color1)

    pyplot.plot(input_x, inputs1[selected_example, :], '.-', color=color2)
    pyplot.plot(prediction_x, output1[selected_example, :], color='k', linewidth=3, alpha=0.5)
    pyplot.plot(prediction_x, predictions1[selected_example, :], '--', color=color2)

    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Input/Output/Target')

pyplot.subplot(1, 4, test_condition + 2)
L = network_model.layers[0]
weights = L.get_weights()[0]
max = numpy.max(numpy.abs(weights))
pyplot.imshow(weights, vmin=-max, vmax=max)
pyplot.colorbar()

pyplot.tight_layout()
pyplot.show()