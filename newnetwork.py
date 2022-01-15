import keras
import numpy
from matplotlib import pyplot
import Misc

input_duration = 0.25
input_n = int(input_duration * 100) - 1

full_model = keras.Sequential()
input_layer = keras.layers.Dense(units=50, input_shape=(input_n * 2,), activation='tanh')
hidden_layer1 = keras.layers.Dense(50, activation='tanh')
hidden_layer2 = keras.layers.Dense(50, activation='tanh')
output_layer = keras.layers.Dense(2, activation='tanh')

full_model.add(input_layer)
full_model.add(hidden_layer1)
full_model.add(hidden_layer2)
full_model.add(output_layer)

loss = keras.losses.MeanSquaredError()
full_model.compile('adam', loss=loss)

inputs = []
outputs = []

disturb = True
examples = 500

for i in range(examples):
    history = Misc.waves(duration=input_duration, offset=90)
    output = history[-1, :]
    input = history[:-1, :]
    input = input.flatten(order='F')
    inputs.append(input)
    outputs.append(output)

inputs = numpy.array(inputs)
outputs = numpy.array(outputs)

if disturb:
    inputs[:, 0:24] = 0
    outputs[:, 0] = 0

training_history = full_model.fit(inputs, outputs, epochs=500)



#%%


prediction = full_model.predict(inputs)
error0 = numpy.mean(((prediction[:, 0] - outputs[:, 0]) ** 2))
error1 = numpy.mean(((prediction[:, 1] - outputs[:, 1]) ** 2))
print(error0, error1)

pyplot.subplot(1, 2, 1)
pyplot.scatter(prediction[:, 0], outputs[:, 0])
pyplot.subplot(1, 2, 2)
pyplot.scatter(prediction[:, 1], outputs[:, 1])
pyplot.show()

pyplot.scatter(outputs[:, 0], outputs[:, 1])
pyplot.show()
