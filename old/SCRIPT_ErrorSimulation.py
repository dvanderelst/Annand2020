import numpy
from matplotlib import pyplot
from numpy import random

from library import Misc

fs = 10
duration = 4

delays = []
for i in range(2500):
    n = duration * fs
    angles = numpy.random.random(n) * 2 * numpy.pi

    x_noise = numpy.cos(angles) * 0.05
    y_noise = numpy.sin(angles) * 0.05

    mx_corr, rms, delay, corr = Misc.lag_finder(x_noise, y_noise, dt=1 / fs, plot=False)
    delays.append(delay)

# %%

t, x_signal = Misc.wave(duration=duration, fs=fs, phase=0)
t, y_signal = Misc.wave(duration=duration, fs=fs, phase=90)

mx_corr, rms, delay, corr = Misc.lag_finder(x_noise, y_noise, dt=1 / fs, plot=True)

pyplot.subplot(1,2,1)
pyplot.plot(x_noise, y_noise, markersize=6, marker='o', linestyle='dashed')
pyplot.xlabel('dx')
pyplot.ylabel('dy')
ax = pyplot.gca()
ax.set_aspect('equal', 'box')

pyplot.subplot(1,2,2)
pyplot.plot(x_signal + x_noise, y_signal + y_noise, markersize=1, marker='o', linestyle='dashed')
pyplot.xlabel('X + noise')
pyplot.ylabel('Y + noise')
ax = pyplot.gca()
ax.set_aspect('equal', 'box')
pyplot.show()

pyplot.plot(x_noise)
pyplot.plot(y_noise)
pyplot.show()

pyplot.figure()
pyplot.hist(delays)
pyplot.show()
print(numpy.mean(delays))
