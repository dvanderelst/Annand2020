import numpy
from matplotlib import pyplot
from scipy import fft
from scipy import signal
from scipy.signal import spectrogram
from library import Misc
from library import Correlation

#######################################
fs = 100
overlap_pct = 99
segment_sec = 15
max_freq = 1
min_energy_pct = 10

data1 = Misc.read_csv('digitize/trial1.csv', 0, 60, fs=fs)
data5 = Misc.read_csv('digitize/trial5.csv', 0, 60, fs=fs)
data10 = Misc.read_csv('digitize/trial10.csv', 0, 60, fs=fs)
data15 = Misc.read_csv('digitize/trial15.csv', 0, 60, fs=fs)
data20 = Misc.read_csv('digitize/trial20.csv', 0, 60, fs=fs)
ind = Misc.read_csv('digitize/ind.csv', 20, 40, fs=fs)
dyad = Misc.read_csv('digitize/dyad.csv', 10, 50, fs=fs)
explore = Misc.read_csv('digitize/explore.csv', 0, 120, fs=fs)

sets = [data1, data5, data10, data15, data20, ind, dyad, explore]
set_names = ['Trial 1', 'Trial 5', 'Trial 10', 'Trial 15', 'Trial 20', 'ind', 'dyad', 'explore']

phase_differences = []
correlations_amp = []
correlations_pha = []
time_shifts = []
peak_freqs = []

pyplot.close('all')

for data in sets:
    x = data.x
    y = data.y
    x = x - numpy.mean(x)
    y = y - numpy.mean(y)

    nperseg = fs * segment_sec
    noverlap = int(nperseg * (overlap_pct / 100))

    _, _, psd_x = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='psd')
    _, _, angle_x = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='phase')

    _, _, psd_y = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='psd')
    freqs, t, angle_y = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='phase')

    shift = Correlation.lag_finder(x, y, fs=fs)
    time_shifts.append(shift)

    freq_selection = freqs < max_freq

    selected_freqs = freqs[freq_selection]

    summed = numpy.mean(psd_x + psd_y, axis=1) ** 2
    summed = summed / numpy.max(summed)
    max_i = numpy.argmax(summed)
    peak_f = freqs[max_i]
    peak_freqs.append(peak_f)

    angle_analysis_band = summed > (min_energy_pct / 100)

    psd_x = psd_x[freq_selection, :]
    psd_y = psd_y[freq_selection, :]

    angle_x = angle_x[angle_analysis_band, :]
    angle_y = angle_y[angle_analysis_band, :]

    angle_difference = numpy.degrees(angle_y - angle_x) % 360
    if len(angle_difference.shape) > 1: angle_difference = numpy.mean(angle_difference, axis=0)

    pyplot.figure(figsize=(12, 3))
    pyplot.subplot(1, 4, 1)
    pyplot.plot(data.t, x)
    pyplot.plot(data.t, y)
    pyplot.legend(['X', 'Y'])

    pyplot.subplot(1, 4, 2)
    pyplot.plot(angle_difference)
    pyplot.ylabel('Delta Angle')
    pyplot.xlabel('time, sec.')
    pyplot.hlines(270, 0, len(angle_difference), linestyles='--')
    pyplot.hlines(180, 0, len(angle_difference), linestyles='--')
    pyplot.ylim(0, 360)

    extent = [min(t), max(t), min(selected_freqs), max(selected_freqs)]

    pyplot.subplot(1, 4, 3)
    pyplot.imshow(psd_x, aspect='auto', extent=extent, origin='lower')
    # pyplot.contourf(t, selected_freqs, psd_x, cmap='jet')
    pyplot.hlines(0.19, min(t), max(t))
    pyplot.ylabel('Frequency')
    pyplot.xlabel('time, sec.')

    pyplot.subplot(1, 4, 4)
    pyplot.imshow(psd_y, aspect='auto', extent=extent, origin='lower')
    # pyplot.contourf(t, selected_freqs, psd_y, cmap='jet')
    pyplot.hlines(0.19, min(t), max(t))
    pyplot.ylabel('Frequency')
    pyplot.xlabel('time, sec.')

    pyplot.tight_layout()
    pyplot.show()

    angles_x = numpy.sin(angle_x)
    angles_y = numpy.cos(angle_y)

    # pyplot.figure()
    # pyplot.subplot(1,2,1)
    # pyplot.imshow(angles_x, aspect='auto')
    # pyplot.subplot(1,2,2)
    # pyplot.imshow(angles_y, aspect='auto')
    # pyplot.show()

    r = numpy.corrcoef(psd_x.flatten(), psd_y.flatten())
    r = r[0, 1]
    correlations_amp.append(r)

    r = numpy.corrcoef(angles_x.flatten(), angles_y.flatten())
    r = r[0, 1]
    correlations_pha.append(r)

    phase_differences.append(angle_difference)

peak_freqs = numpy.array(peak_freqs)
time_shifts = numpy.array(time_shifts)
phase_shifts = 360 * (time_shifts / (1/peak_freqs))
# %%
pyplot.figure(figsize=(6,3))
pyplot.plot([1, 5, 10, 15, 20], correlations_amp[:-3], marker='o', color='k')
pyplot.plot([1, 5, 10, 15, 20], correlations_pha[:-3], marker='s', color='k')
pyplot.plot([19.5], correlations_amp[-3], 'ro', alpha=0.5)
pyplot.plot([20.5], correlations_amp[-2], 'go', alpha=0.5)
pyplot.plot([21], correlations_amp[-1], 'bo', alpha=0.5)

pyplot.plot([19.5], correlations_pha[-3], 'rs', alpha=0.5)
pyplot.plot([20.5], correlations_pha[-2], 'gs', alpha=0.5)
pyplot.plot([21], correlations_pha[-1], 'bs', alpha=0.5)

pyplot.legend(['Amplitude','Phase', 'High RMSE ind.', 'High RMSE dyad', 'Exploring ind.'])
pyplot.xticks([1, 5, 10, 15, 20])
pyplot.title('Spectrogram Correlation')
pyplot.xlabel('Trial')
pyplot.ylabel('Pearson Corr.')
pyplot.grid()
pyplot.ylim(0.7, 1.1)
pyplot.tight_layout()
pyplot.show()
