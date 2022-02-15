import numpy
from matplotlib import pyplot
from numpy import random

from library import Misc

fs = 12
duration = 10
n = duration * fs

time_shift_samples = 3
coupling_angle = 90

angles1 = numpy.random.random(n) * 2 * numpy.pi
angles2 = numpy.roll(angles1, 3) + numpy.radians(coupling_angle)

x_noise = numpy.sin(angles1) * 0.05
y_noise = numpy.sin(angles2) * 0.05

mx_corr, rms, delay, corr = Misc.lag_finder(x_noise, y_noise, dt=1 / fs, plot=False)

pyplot.plot(x_noise)
pyplot.plot(y_noise)
pyplot.show()

pyplot.plot(x_noise,y_noise)
pyplot.show()

print(delay)