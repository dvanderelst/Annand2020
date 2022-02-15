import numpy
import scipy.stats
from matplotlib.lines import Line2D
from library import Misc
from library import CrossCorr
from matplotlib import pyplot
from scipy.signal import hilbert
from scipy import signal
from scipy.signal.windows import hamming

def lag_finder(y1, y2, dt=0.1, plot=False):
    n = len(y1)
    sr = 1 / dt

    corr = signal.correlate(y2, y1, mode='same') / numpy.sqrt(signal.correlate(y1, y1, mode='same')[int(n / 2)] * signal.correlate(y2, y2, mode='same')[int(n / 2)])

    delay_arr = numpy.linspace(-0.5 * n / sr, 0.5 * n / sr, n)
    delay = delay_arr[numpy.argmax(corr)]
    print('y2 is ' + str(delay) + ' behind y1')

    rms = numpy.mean((y1 ** 2 + y2 ** 2) ** 0.5)
    mx_corr = numpy.max(numpy.abs(corr))

    if plot:
        pyplot.figure()
        pyplot.plot(delay_arr, corr)
        pyplot.title('Lag: ' + str(numpy.round(delay, 3)) + ' s')
        pyplot.xlabel('Lag')
        pyplot.ylabel('Correlation coeff')
        pyplot.show()

    return mx_corr, rms, delay, corr

def signal_ramp(n, percent):
    if percent > 49: percent = 49
    length = int(numpy.floor((n * percent) / 100))
    window = hamming(length * 2 + 1)
    window = window - numpy.min(window)
    window = window / numpy.max(window)
    left = window[0:length + 1]
    right = window[length:]
    buffer = numpy.ones(n - 2 * left.size)
    total = numpy.hstack((left, buffer, right))
    return total

def cross_correlate(y1, y2):

    #y1 = y1 - numpy.mean(y1)
    #y2 = y2 - numpy.mean(y2)

    #y1 = y1 - 1
    #y2 = y2 - 1



    # xcorr = signal.correlate(y2, y1, mode='same')
    # auto1 = numpy.max(signal.correlate(y1, y1, mode='same'))
    # auto2 = numpy.max(signal.correlate(y2, y2, mode='same'))
    #
    #
    # corr = xcorr / (auto1 * auto2) ** 0.5

    corr = scipy.stats.pearsonr(y1, y2)
    corr = corr[0]
    corr = numpy.array([corr])


    return corr

def short_term_xcorr(d1,d2):
    n = len(d1)
    start = 0
    end = 60
    shift = 1
    correlations = []
    while end < n:
        print(start, end)
        start = start + shift
        end = end+shift
        corr = scipy.stats.pearsonr(d1[start:end], d2[start:end])
        c = abs(corr[0])
        correlations.append(c)
    correlations = numpy.array(correlations)
    correlations = numpy.mean(correlations)
    return correlations

data1 = Misc.read_csv('digitize/trial1.csv', 0, 60)
data5 = Misc.read_csv('digitize/trial5.csv', 0, 60)
data10 = Misc.read_csv('digitize/trial10.csv', 0, 60)
data15 = Misc.read_csv('digitize/trial15.csv', 0, 60)
data20 = Misc.read_csv('digitize/trial20.csv', 0, 60)
ind = Misc.read_csv('digitize/ind.csv', 20, 40)
dyad = Misc.read_csv('digitize/dyad.csv', 10, 50)


#%%
fs = 10

sets = [data1, data5, data10, data15, data20, ind, dyad]
set_names = ['data1', 'data5', 'data10', 'data15', 'data20', 'ind, dyad']

shifts = []
xcorrs_amp = []
xcorrs_pha = []
delays = []

for set in sets:
    signal_x = set.x
    signal_y = set.y

    window = signal_ramp(len(signal_x), 5)

    signal_x= signal_x * window
    signal_y = signal_y * window

    analytic_signal_x = hilbert(signal_x)
    amplitude_envelope_x = numpy.abs(analytic_signal_x)
    instantaneous_phase_x = numpy.unwrap(numpy.angle(analytic_signal_x))
    instantaneous_frequency_x = (numpy.diff(instantaneous_phase_x) / (2.0 * numpy.pi) * fs)

    analytic_signal_y = hilbert(signal_y)
    amplitude_envelope_y = numpy.abs(analytic_signal_y)
    instantaneous_phase_y = numpy.unwrap(numpy.angle(analytic_signal_y))
    instantaneous_frequency_y = (numpy.diff(instantaneous_phase_y) / (2.0 * numpy.pi) * fs)

    mx_corr, rms, delay, corr = lag_finder(signal_x-1, signal_y-1, dt=0.1, plot=False)

    # pyplot.figure()
    # pyplot.subplot(4,1,1)
    # pyplot.plot(signal_x)
    # pyplot.plot(signal_y)
    # pyplot.subplot(4,1,2)
    # pyplot.plot(amplitude_envelope_x)
    # pyplot.plot(amplitude_envelope_y)
    # pyplot.subplot(4,1,3)
    # pyplot.plot(instantaneous_phase_x)
    # pyplot.plot(instantaneous_phase_y)
    # pyplot.subplot(4,1,4)
    # pyplot.plot(instantaneous_frequency_x)
    # pyplot.plot(instantaneous_frequency_y)
    # pyplot.show()

    pyplot.figure()
    difference = numpy.degrees(instantaneous_phase_x - instantaneous_phase_y)
    average_difference = numpy.mean(difference)
    if average_difference < 0: average_difference = 360 + average_difference
    #pyplot.plot(difference)
    #pyplot.show()

    xcorr_amp = short_term_xcorr(amplitude_envelope_x, amplitude_envelope_y)
    xcorr_pha = short_term_xcorr(instantaneous_phase_x, instantaneous_phase_y)
    average_xcorr_amp = numpy.mean(xcorr_amp)
    average_xcorr_pha = numpy.mean(xcorr_pha)


    shifts.append(average_difference)
    xcorrs_amp.append(average_xcorr_amp)
    xcorrs_pha.append(average_xcorr_pha)
    delays.append(delay)

#%%

pyplot.figure()
pyplot.plot(xcorrs_amp)
pyplot.plot(xcorrs_pha)
pyplot.show()