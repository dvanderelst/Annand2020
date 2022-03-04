import numpy
import scipy.stats
from matplotlib import pyplot
from scipy import signal
from scipy.signal.windows import hamming


def lag_finder(y1, y2, fs=10, plot=False):
    n = len(y1)
    corr = signal.correlate(y2, y1, mode='same') / numpy.sqrt(signal.correlate(y1, y1, mode='same')[int(n / 2)] * signal.correlate(y2, y2, mode='same')[int(n / 2)])
    delay_arr = numpy.linspace(-0.5 * n / fs, 0.5 * n / fs, n)
    delay = delay_arr[numpy.argmax(corr)]
    rms = numpy.mean((y1 ** 2 + y2 ** 2) ** 0.5)
    mx_corr = numpy.max(numpy.abs(corr))

    if plot:
        pyplot.figure()
        pyplot.plot(delay_arr, corr)
        pyplot.title('Lag: ' + str(numpy.round(delay, 3)) + ' s')
        pyplot.xlabel('Lag')
        pyplot.ylabel('Correlation coeff')
        pyplot.show()
    return delay





