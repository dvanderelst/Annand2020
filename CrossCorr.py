from scipy import signal
import numpy

def cross_correlate(y1, y2):
    n1 = len(y1)
    n2 = len(y2)

    xcorr = signal.correlate(y2, y1, mode='same')
    auto1 = signal.correlate(y1, y1, mode='same')[int(n1 / 2)]
    auto2 = signal.correlate(y2, y2, mode='same')[int(n2 / 2)]

    corr = xcorr / (auto1 * auto2) ** 0.5
    return corr


def short_term_xcorr(y1,y2):
    n = len(y1)
    start = 0
    end = 30
    shift = 10
    correlations = []
    while end < n:
        print(start, end)
        start = start + shift
        end = end+shift

        c = cross_correlate(y1[start:end], y2[start:end])
        c = numpy.max(numpy.abs(c))

        correlations.append(c)
    correlations = numpy.array(correlations)
    correlations = numpy.mean(correlations)
    return correlations


def short_term_rms(y1,y2):
    n = len(y1)
    start = 0
    end = 30
    shift = 10
    rms = []
    while end < n:
        print(start, end)
        start = start + shift
        end = end+shift

        c = (y1[start:end]**2 + y2[start:end]**2)**0.5
        c = numpy.mean(c)

        rms.append(c)
    correlations = numpy.array(rms)
    correlations = numpy.mean(correlations)
    return correlations