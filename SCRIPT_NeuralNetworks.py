import numpy

import Misc
from matplotlib import pyplot

model = Misc.my_model()

# %%

for trials in range(125):
    waves = Misc.waves(duration=3)
    n = waves.shape[0]
    targets = waves * 1.0
    scrambled = Misc.scramble(waves[:, 0])
    #waves[:,0] = scrambled * 0
    history = model.fit(waves, targets, epochs=1)

prediction = model.predict(waves)

c1 = numpy.corrcoef(prediction[:, 0], waves[:, 0])
c1 = c1[0, 1]

c2 = numpy.corrcoef(prediction[:, 1], waves[:, 1])
c2 = c2[0, 1]

#%%
pyplot.figure(figsize=(10, 5))

pyplot.subplot(2, 2, 1)
pyplot.plot(waves)
pyplot.title('Input (Selected example)')
pyplot.subplot(2, 2, 2)
pyplot.plot(targets)
pyplot.title('Target (Selected example)')
pyplot.subplot(2, 2, 3)
pyplot.plot(prediction)
pyplot.title('Learned output (Selected example)')
pyplot.tight_layout()
pyplot.subplot(2, 2, 4)
pyplot.title('Spatial representation (Selected example)')
time = numpy.arange(n)
pyplot.scatter(prediction[:,0],prediction[:,1], c=time, s=5, cmap='hot')
pyplot.xlim(-0.5,0.5)
pyplot.ylim(-0.5,0.5)
pyplot.show()


print(c1, c2)

# full_model.add(output_layer)
# full_model.add(input_layer)
# for nodes in layers[1:]: full_model.add(keras.layers.Dense(nodes, activation='relu'))
# output_layer = keras.layers.Dense(output_shape, activation='softmax')
# full_model.add(output_layer)
#
# #%%
# loss = keras.losses.CategoricalCrossentropy()
# full_model.compile('adam', loss=loss)
#
# for i in range(repeats):
#     noise = numpy.random.normal(0, noise_level, input.shape)
#     noisy_input = input + noise
#     history = full_model.fit(noisy_input, encoded, epochs=50)
#     loss = history.history['loss']
#
#
#
#  prediction = full_model.predict(noisy_input)
