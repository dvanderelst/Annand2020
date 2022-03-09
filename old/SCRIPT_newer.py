import numpy
from matplotlib import pyplot
from scipy import fft
from scipy import signal
from scipy.signal import spectrogram
from scipy.signal import hilbert
from scipy.signal import periodogram
from scipy.signal import welch
import pandas

from library import Misc

def analyze(signal, fs):
    signal = signal - numpy.mean(signal)
    analytic_signal = hilbert(signal)
    amplitude_envelope = numpy.abs(analytic_signal)
    instantaneous_phase = numpy.unwrap(numpy.angle(analytic_signal))
    instantaneous_frequency = (numpy.diff(instantaneous_phase) / (2.0*numpy.pi) * fs)
    return amplitude_envelope, instantaneous_phase, instantaneous_frequency

def spectrum(data, fs):
    x = data.dx.values
    y = data.dy.values
    fx, vx = my_fft(x, fs=fs)
    fy, vy = my_fft(y, fs=fs)
    selected = fx < 0.5
    fx = fx[selected]
    vx = vx[selected]
    vy = vy[selected]
    pyplot.figure()
    pyplot.plot(fx, numpy.abs(vx))
    pyplot.plot(fx, numpy.abs(vy))
    pyplot.show()
    return fx, vx, vy


def my_fft(x, fs):
    x = x.flatten()
    x = x - numpy.mean(x)
    spectrum = fft(x)
    length = int(spectrum.size/2)
    positive = spectrum[0:length]
    frequencies = numpy.linspace(0, fs/2, length)
    return frequencies[1:], positive[1:]

#######################################
fs = 25

data1 = Misc.read_csv('digitize/trial1.csv', 0, 60, fs=fs)
data5 = Misc.read_csv('digitize/trial5.csv', 0, 60, fs=fs)
data10 = Misc.read_csv('digitize/trial10.csv', 0, 60, fs=fs)
data15 = Misc.read_csv('digitize/trial15.csv', 0, 60, fs=fs)
data20 = Misc.read_csv('digitize/trial20.csv', 0, 60, fs=fs)

ind = Misc.read_csv('digitize/ind.csv', 20, 40, fs=fs)
dyad = Misc.read_csv('digitize/dyad.csv', 10, 50, fs=fs)



f, vx, vy = spectrum(data20, fs=fs)
angle_x = numpy.angle(vx)
angle_y = numpy.angle(vy)
diff = numpy.degrees(angle_x - angle_y)
#diff[diff < -180] = 360 + diff[diff < -180]
#diff[diff > 180] = 360 - diff[diff > 180]
vx = numpy.abs(vx)
vy = numpy.abs(vy)
result = {'f':f,'x':vx, 'y':vy, 'd': diff}
result = pandas.DataFrame(result)

s = numpy.sort(vx)
s = s[::-1]
ss = numpy.cumsum(s)
ss = ss/numpy.max(ss)

pyplot.figure()
pyplot.scatter(vx, vy)
pyplot.show()