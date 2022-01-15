import keras
import numpy
from matplotlib import pyplot
import tensorflow as tf
import Misc



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class weight_constraint(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be centered around `ref_value`."""

  def __init__(self):
      pass


  def __call__(self, w):

      s = w.shape.as_list()
      n = s[0]
      m = numpy.zeros((n, n))
      for r in range(n):
          for c in range(n):
              if c > (n / 2) - 1 and r < (n / 2): m[r, c] = 1
              if c < (n / 2) and r > (n / 2) - 1: m[r, c] = 1
      m = m + 1
      indices = numpy.nonzero(m)
      indices = numpy.ravel_multi_index(indices, dims=s, order='C')
      indices = indices.tolist()
      tf.scatter_nd([w], [1,2,3], [0])
      tf.print(w)

      return w


  def get_config(self):
    return {'ref_value': self.ref_value}


def create_examples(training_condition):
    n = 360
    inputs = []
    outputs = []
    for i in range(n):
        if training_condition == 0: offset = 90
        if training_condition == 1: offset = i
        waves = Misc.waves(frequency=3, duration=0.251, phase0=i, offset=offset)
        input0 = waves[0:13, 0]
        output0 = waves[13:, 0]

        input1 = waves[0:13, 1]
        output1 = waves[13:, 1]

        input = numpy.concatenate((input0, input1))
        output = numpy.concatenate((output0, output1))

        inputs.append(input)
        outputs.append(output)

    inputs = numpy.array(inputs)
    outputs = numpy.array(outputs)
    return inputs, outputs


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pyplot.rcParams['text.usetex'] = True
color1 = '#fc8d59'
color2 = '#91bfdb'

full_model = keras.Sequential()
input_layer = keras.layers.Dense(units=26, input_shape=(26,),
                                 activation='tanh',
                                 kernel_initializer='zeros',
                                 bias_initializer='zeros',
                                 kernel_constraint=weight_constraint())
#hidden_layer1 = keras.layers.Dense(100, activation='tanh')
# hidden_layer2 = keras.layers.Dense(75, activation='tanh')
# hidden_layer3 = keras.layers.Dense(50, activation='tanh')
#output_layer = keras.layers.Dense(26, activation='tanh')

full_model.add(input_layer)
#full_model.add(hidden_layer1)
# full_model.add(hidden_layer2)
# full_model.add(hidden_layer3)
#full_model.add(output_layer)

loss = keras.losses.MeanSquaredError()
full_model.compile('adam', loss=loss)


training_condition = 0
examples = 360

inputs, outputs = create_examples(training_condition)

training_history = full_model.fit(inputs, outputs, epochs=50)


# %%
inputs, outputs = create_examples(0)
selected_example = 250
pyplot.figure(figsize=(9,3))

for test_condition in [0, 1, 2]:
    test_inputs = inputs * 1.0

    size = test_inputs[:, 0:13]
    size = size.shape
    noise = ((numpy.random.random(size) * 2) - 1)
    noise = noise

    if test_condition == 1: test_inputs[:, 0:13] = noise #test_inputs[:, 0:13] * 0
    if test_condition == 2: test_inputs[:, 13:] = noise #test_inputs[:, 13:] * 0


    prediction = full_model.predict(test_inputs)

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
L = full_model.layers[0]
weights = L.get_weights()[0]
max = numpy.max(numpy.abs(weights))
pyplot.imshow(weights, vmin=-max, vmax=max)
pyplot.colorbar()


pyplot.tight_layout()
pyplot.show()

#
# #%%
#
#
# prediction = full_model.predict(inputs)
# error0 = numpy.mean(((prediction[:, 0] - outputs[:, 0]) ** 2))
# error1 = numpy.mean(((prediction[:, 1] - outputs[:, 1]) ** 2))
# print(error0, error1)
#
# pyplot.subplot(1, 2, 1)
# pyplot.scatter(prediction[:, 0], outputs[:, 0])
# pyplot.subplot(1, 2, 2)
# pyplot.scatter(prediction[:, 1], outputs[:, 1])
# pyplot.show()
#
# pyplot.scatter(outputs[:, 0], outputs[:, 1])
# pyplot.show()
#%%
