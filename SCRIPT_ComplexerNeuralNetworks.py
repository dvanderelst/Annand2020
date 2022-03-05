import keras
import numpy
from matplotlib import pyplot
from library import Correction
from library import Misc
from library import NetworkFunctions
from os import path
from library import Settings

pyplot.rcParams['text.usetex'] = True

color1 = '#fc8d59'
color2 = '#91bfdb'
apply_constraints = False
training_x_noise = False
training_examples = 500
loss = keras.losses.MeanSquaredError()
historic_samples = 15
predicted_samples = 15

output_name = 'coupled'

# %% Train Network

network_model = keras.Sequential()
constraint = NetworkFunctions.weight_constraint(apply_constraints)
input_layer = keras.layers.Dense(units=predicted_samples * 2, input_shape=(historic_samples * 2,), activation='tanh', kernel_constraint=constraint)
network_model.add(input_layer)
network_model.compile('adam', loss=loss)

inputs, outputs = Misc.create_training_examples(frequency=1, input_samples=historic_samples, output_samples=predicted_samples, n=training_examples)
noise = ((numpy.random.random((training_examples, historic_samples)) * 2) - 1)
if training_x_noise: inputs[:,0:15] = noise

training_history = network_model.fit(inputs, outputs, epochs=500)

# %% Visualise output

selected_example = 250
pyplot.figure(figsize=(10, 2))

labels = ['a', 'b', 'c']

for test_condition in [0, 1, 2]:
    test_inputs = inputs * 1.0

    size = test_inputs[:, 0:historic_samples]
    size = size.shape
    noise = ((numpy.random.random(size) * 2) - 1)
    noise = (noise / numpy.max(numpy.abs(noise))) * 0.5

    title = 'Example Inputs: $x_{T,i}$, $y_{T,i}$'
    if test_condition == 1:
        test_inputs[:, 0:historic_samples] = noise
        title =  'Example Inputs: noise,  $y_{T,i}$'
    if test_condition == 2:
        test_inputs[:, historic_samples:] = noise
        title = 'Example Inputs: $x_{T,i}$, noise'

    prediction = network_model.predict(test_inputs)
    # Feed the prediction to a proportional controller
    prediction = Correction.run_corrections(prediction, predicted_samples=predicted_samples)

    inputs0 = test_inputs[:, 0:historic_samples]
    inputs1 = test_inputs[:, historic_samples:]
    predictions0 = prediction[:, 0:predicted_samples]
    predictions1 = prediction[:, predicted_samples:]

    output0 = outputs[:, 0:predicted_samples]
    output1 = outputs[:, predicted_samples:]

    input_x = numpy.arange(historic_samples) * 0.1
    prediction_x = (numpy.arange(predicted_samples) + historic_samples) * 0.1

    pyplot.subplot(1, 4, test_condition + 1)
    pyplot.plot(input_x, inputs0[selected_example, :], '-', color=color1)
    pyplot.plot(prediction_x, output0[selected_example, :], color='k', linewidth=3, alpha=0.5)
    pyplot.plot(prediction_x, predictions0[selected_example, :], '--', color=color1)

    pyplot.plot(input_x, inputs1[selected_example, :], '-', color=color2)
    pyplot.plot(prediction_x, output1[selected_example, :], color='k', linewidth=3, alpha=0.5)
    pyplot.plot(prediction_x, predictions1[selected_example, :], '--', color=color2)

    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Input/Output/Target')
    pyplot.title(title)
    Misc.label(labels[test_condition], 0.05, 0.9)

pyplot.subplot(1, 4, test_condition + 2)
L = network_model.layers[0]
weights = L.get_weights()[0]
max = numpy.max(numpy.abs(weights))
pyplot.imshow(weights, vmin=-max, vmax=max, cmap='Spectral')
pyplot.title('Network weights')
pyplot.xlabel('Input unit nr')
pyplot.ylabel('Output unit nr')
pyplot.colorbar()
Misc.label('d', 0.1, 0.9)
pyplot.tight_layout()
image_output = path.join(Settings.image_folder, output_name + '.pdf')
pyplot.savefig(image_output)
pyplot.show()
