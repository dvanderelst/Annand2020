import numpy
from matplotlib import pyplot
import keras
from keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas
from scipy.interpolate import interp1d
from scipy import signal





def my_model():
    full_model = keras.Sequential()
    input_layer = keras.layers.Dense(units=2, input_shape=(2,), activation='tanh')
    hidden_layer = keras.layers.Dense(2, activation='tanh')
    output_layer = keras.layers.Dense(2, activation='tanh')

    full_model.add(input_layer)
    # full_model.add(hidden_layer)
    #full_model.add(output_layer)

    loss = keras.losses.MeanSquaredError()
    full_model.compile('RMSprop', loss=loss)

    return full_model


def wave(frequency=1, duration=10, fs=100, phase=0, plot=False):
    p = numpy.deg2rad(phase)
    points = numpy.arange(start=0, stop=duration, step=1 / fs)
    signal = 0.5 * numpy.sin(2 * numpy.pi * frequency * points + p)

    if plot:
        pyplot.plot(points, signal)
        pyplot.grid()
        pyplot.show()
    return points, signal


def waves(duration=10, offset=90):
    phase1 = numpy.random.randint(0, 360)
    p1, s1 = wave(duration=duration, phase=phase1)
    p2, s2 = wave(duration=duration, phase=phase1 + offset)
    w = [s1, s2]
    w = numpy.array(w)
    w = numpy.transpose(w)
    return w


def scramble(array):
    array = array * 1.0
    numpy.random.shuffle(array)
    return array


def read_csv(filename, start, end):
    data = pandas.read_csv(filename)
    time_steps = numpy.arange(start, end, 0.1)

    f1 = interp1d(data.x, data.Curve1, fill_value="extrapolate")
    Curve1 = f1(time_steps)

    f2 = interp1d(data.x, data.Curve2, fill_value="extrapolate")
    Curve2 = f2(time_steps)

    f3 = interp1d(data.x, data.Curve3, fill_value="extrapolate")
    Curve3 = f3(time_steps)

    f4 = interp1d(data.x, data.Curve4, fill_value="extrapolate")
    Curve4 = f4(time_steps)

    result = {'t': time_steps, 'x': Curve1, 'y': Curve2, 'tx': Curve3, 'ty': Curve4}
    result = pandas.DataFrame(result)
    result['dx'] = result.x - result.tx
    result['dy'] = result.y - result.ty

    return result


def plot_data(data):
    pyplot.figure()
    pyplot.plot(data.t, data.x, 'b')
    pyplot.plot(data.t, data.y, 'r')
    pyplot.plot(data.t, data.tx, '--b')
    pyplot.plot(data.t, data.ty, '--r')
    pyplot.show()


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

# p1,s1 = wave(phase=0, duration=3)
# p2,s2 = wave(phase=90, duration=3)
#
# s3 = scramble(s1)
#
# pyplot.plot(s1)
# pyplot.plot(s2)
# pyplot.plot(s3)
# pyplot.show()
