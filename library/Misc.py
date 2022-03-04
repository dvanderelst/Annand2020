import numpy
from matplotlib import pyplot
import keras
from keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas
from scipy.interpolate import interp1d
from scipy import signal
from scipy.signal import windows
from scipy.signal.windows import hamming


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


def wave(frequency=1, duration=10.0, fs=10, phase=0, plot=False):
    p = numpy.deg2rad(phase)
    points = numpy.arange(start=0, stop=duration, step=1 / fs)
    signal = 0.5 * numpy.sin(2 * numpy.pi * frequency * points + p)

    if plot:
        pyplot.plot(points, signal)
        pyplot.grid()
        pyplot.show()
    return points, signal


def waves(frequency=1,duration=10.0, phase0=False, offset=90):
    if not phase0: phase0 = numpy.random.randint(0, 360)
    p1, s1 = wave(frequency=frequency, duration=duration, phase=phase0)
    p2, s2 = wave(frequency=frequency, duration=duration, phase=phase0 + offset)
    w = [s1, s2]
    w = numpy.array(w)
    w = numpy.transpose(w)
    return w


def scramble(array):
    array = array * 1.0
    numpy.random.shuffle(array)
    return array


def read_csv(filename, start, end, fs=10):
    data = pandas.read_csv(filename)
    time_steps = numpy.arange(start, end, 1/fs)

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

    dx = result.x - result.tx
    dy = result.y - result.ty

    result['dx'] = dx
    result['dy'] = dy

    dx = dx - numpy.mean(dx)
    dy = dy - numpy.mean(dy)
    dx = dx / numpy.max(numpy.abs(dx))
    dy = dy / numpy.max(numpy.abs(dy))

    radians_x = numpy.arccos(dx)
    derivative = numpy.gradient(dx)
    radians_x[derivative > 0] = - radians_x[derivative > 0]
    #radians_x = numpy.unwrap(radians_x)

    radians_y = numpy.arccos(dy)
    derivative = numpy.gradient(dy)
    radians_y[derivative > 0] = - radians_y[derivative > 0]
    #radians_y = numpy.unwrap(radians_y)

    result['radians_x'] = radians_x
    result['radians_y'] = radians_y



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



def smooth_signal(signal, samples, window='box'):
    if window == 'box': w = windows.boxcar(samples)
    if window == 'han': w = windows.hann(samples)
    if window == 'flat': w = windows.flattop(samples)
    w = w / numpy.sum(w)
    smoothed = numpy.convolve(signal, w, mode='same')
    return smoothed



def create_training_examples(frequency=3, input_samples=15, output_samples=15 , n = 250):
    inputs = []
    outputs = []
    total_samples = input_samples + output_samples
    required_duration = 1 + int(total_samples / 10)
    for i in range(n):
        data = waves(frequency=frequency, duration=required_duration, phase0=False, offset=90)
        input0 = data[0:input_samples, 0]
        output0 = data[input_samples:total_samples, 0]

        input1 = data[0:input_samples, 1]
        output1 = data[input_samples:total_samples:, 1]

        input = numpy.concatenate((input0, input1))
        output = numpy.concatenate((output0, output1))

        inputs.append(input)
        outputs.append(output)

    inputs = numpy.array(inputs)
    outputs = numpy.array(outputs)
    return inputs, outputs

# def create_training_examples(frequency=3, n = 250):
#     inputs = []
#     outputs = []
#     for i in range(n):
#         data = waves(frequency=frequency, duration=0.251, phase0=False, offset=90)
#         input0 = data[0:13, 0]
#         output0 = data[13:, 0]
#
#         input1 = data[0:13, 1]
#         output1 = data[13:, 1]
#
#         input = numpy.concatenate((input0, input1))
#         output = numpy.concatenate((output0, output1))
#
#         inputs.append(input)
#         outputs.append(output)
#
#     inputs = numpy.array(inputs)
#     outputs = numpy.array(outputs)
#     return inputs, outputs
