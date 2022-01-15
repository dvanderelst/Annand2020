import numpy
import tensorflow as tf
import Misc
from matplotlib import pyplot

pyplot.rcParams['text.usetex'] = True
color1 = '#fc8d59'
color2 = '#91bfdb'

model = Misc.my_model()
tf.keras.utils.plot_model(model, to_file='test.png')

condition = 2

if condition == 0:
    output_file = 'latex/neural_network_0.pdf'
    signal_labels = ['$S_1$', '$S_2$']

if condition == 1:
    output_file = 'latex/neural_network_1.pdf'
    signal_labels = ['$S_1$', '$S_3$']

if condition == 2:
    output_file = 'latex/neural_network_2.pdf'
    signal_labels = ['$S_1$', '$S_4$']
# %%

for trials in range(250):
    waves = Misc.waves(duration=3)
    n = waves.shape[0]
    targets = waves * 1.0

    if condition == 1:
        scrambled = Misc.scramble(waves[:, 1])
        waves[:,1] = scrambled

    if condition == 2:
        scrambled = Misc.scramble(waves[:, 1])
        waves[:, 1] = scrambled * 0


    history = model.fit(waves, targets, epochs=1)

L = model.layers[0]
weights = L.get_weights()[0]
prediction = model.predict(waves)

c1 = numpy.corrcoef(prediction[:, 0], waves[:, 0])
c1 = c1[0, 1]

c2 = numpy.corrcoef(prediction[:, 1], waves[:, 1])
c2 = c2[0, 1]

#%%
pyplot.figure(figsize=(9, 3))

pyplot.subplot(1, 3, 1)
pyplot.plot(waves[:,0], color=color1)
pyplot.plot(waves[:,1], color=color2)
pyplot.legend(signal_labels, loc='upper left')
pyplot.title('(a) Input (Selected example)')
pyplot.xlabel('Sample number')
pyplot.yticks([])

pyplot.subplot(1, 3, 2)

pyplot.plot(targets[:,0], color='k', linewidth=5)
pyplot.plot(targets[:,1], color='k', linewidth=5)

pyplot.plot(prediction[:,0], color=color1, linestyle='--')
pyplot.plot(prediction[:,1], color=color2, linestyle='--')

pyplot.title('(b) Learned output (Selected example)')
pyplot.xlabel('Sample number')
pyplot.yticks([])

pyplot.subplot(1, 3, 3)
pyplot.imshow(weights, cmap='bwr',vmin=-1, vmax=1) #%Todo: set cm from -1 to 1
pyplot.colorbar()
pyplot.xticks([0,1], ['To Output 0', 'To Output 1'])
pyplot.yticks([0,1], ['From Input 0', 'From Input 1'])
pyplot.title('(c) Network Weights')
#pyplot.title('Spatial representation (Selected example)')
#time = numpy.arange(n)
#pyplot.scatter(prediction[:,0],prediction[:,1], c=time, s=5, cmap='hot')
#pyplot.xlim(-0.5,0.5)
#pyplot.ylim(-0.5,0.5)
pyplot.tight_layout()
pyplot.savefig(output_file)
pyplot.show()


print(c1, c2)

#%% plot parts
pyplot.figure(figsize=(6,3))
pyplot.subplot(2, 2, 1)
pyplot.plot(waves[:, 0], 'k')
pyplot.xticks([])
pyplot.yticks([])

pyplot.subplot(2, 2, 2)
pyplot.plot(waves[:, 1], 'k')
pyplot.xticks([])
pyplot.yticks([])

pyplot.subplot(2, 2, 3)
scrambled = Misc.scramble(waves[:, 0])
pyplot.plot(scrambled, 'k')
pyplot.xticks([])
pyplot.yticks([])

pyplot.subplot(2, 2, 4)
pyplot.plot(scrambled * 0 , 'k')
pyplot.xticks([])
pyplot.yticks([])


pyplot.tight_layout()
pyplot.savefig('plot_parts.svg')
pyplot.show()

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
