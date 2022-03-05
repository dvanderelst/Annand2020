import numpy
from matplotlib import pyplot
from scipy.signal import spectrogram

from library import Settings
from library import Misc
from os import path

#######################################
fs = 100
overlap_pct = 99
segment_sec = 15
max_freq = 0.5
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
target_freqs = [0.19,0.19,0.19,0.19,0.19,0.38,0.19,0.19]
phase_differences = []
correlations_amp = []
correlations_pha = []
pyplot.close('all')

for data, set_name, target_freq in zip(sets, set_names, target_freqs):
    x = data.x
    y = data.y
    x = x - numpy.mean(x)
    y = y - numpy.mean(y)

    nperseg = fs * segment_sec
    noverlap = int(nperseg * (overlap_pct / 100))

    _, _, psd_x = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='psd')
    _, _, angle_x = spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='phase')

    _, _, psd_y = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='psd')
    freqs, spectrogram_time, angle_y = spectrogram(y, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='phase')
    spectrogram_time = spectrogram_time + min(data.t.values)


    freq_selection = freqs < max_freq

    selected_freqs = freqs[freq_selection]

    summed = numpy.mean(psd_x + psd_y, axis=1) ** 2
    summed = summed / numpy.max(summed)
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
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Displacement')
    pyplot.title(set_name)
    pyplot.legend(['$x_t$', '$y_t$'], loc='lower right')
    Misc.label('a', 0.05, 0.95)


    extent = [min(spectrogram_time), max(spectrogram_time), min(selected_freqs), max(selected_freqs)]


    pyplot.subplot(1, 4, 2)
    pyplot.imshow(psd_x, aspect='auto', extent=extent, origin='lower')
    # pyplot.contourf(t, selected_freqs, psd_x, cmap='jet')
    pyplot.hlines(target_freq, min(spectrogram_time), max(spectrogram_time),colors='red')
    pyplot.ylabel('Frequency')
    pyplot.xlabel('Time (s)')
    pyplot.title('Magnitude spectrogram $x_t$')
    Misc.label('b',  0.05, 0.95, c='white')

    pyplot.subplot(1, 4, 3)
    pyplot.imshow(psd_y, aspect='auto', extent=extent, origin='lower')
    # pyplot.contourf(t, selected_freqs, psd_y, cmap='jet')
    pyplot.hlines(target_freq, min(spectrogram_time), max(spectrogram_time), colors='red')
    pyplot.ylabel('Frequency')
    pyplot.xlabel('time, sec.')
    pyplot.title('Magnitude spectrogram $y_t$')
    Misc.label('c', 0.05, 0.95, c='white')

    pyplot.subplot(1, 4, 4)
    start = numpy.min(spectrogram_time)
    end = numpy.max(spectrogram_time)

    pyplot.plot(spectrogram_time, angle_difference)
    pyplot.ylabel('Phase shift (degrees)')
    pyplot.xlabel('Time (s)')
    pyplot.title('Phase shift')
    Misc.label('d', 0.05, 0.95,)
    pyplot.hlines(270, start, end, linestyles='--')
    pyplot.hlines(180, start, end, linestyles='--')
    pyplot.ylim(0, 360)

    set_name_file = set_name.replace(' ', '_')
    set_name_file = set_name_file.lower()
    image_output = path.join(Settings.image_folder, set_name_file + '.pdf')

    pyplot.tight_layout()
    pyplot.savefig(image_output)
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

# %%
pyplot.figure(figsize=(3, 3))
pyplot.plot([1, 5, 10, 15, 20], correlations_amp[:-3], marker='o', color='k')
pyplot.plot([1, 5, 10, 15, 20], correlations_pha[:-3], marker='s', color='k')
pyplot.plot([19.5], correlations_amp[-3], 'ro', alpha=0.5)
pyplot.plot([20.5], correlations_amp[-2], 'go', alpha=0.5)
pyplot.plot([0], correlations_amp[-1], 'bo', alpha=0.5)

pyplot.plot([19.5], correlations_pha[-3], 'rs', alpha=0.5)
pyplot.plot([20.5], correlations_pha[-2], 'gs', alpha=0.5)
pyplot.plot([0], correlations_pha[-1], 'bs', alpha=0.5)

pyplot.legend(['Amplitude', 'Phase', 'High RMSE ind.', 'High RMSE dyad', 'Exploring ind.'])
pyplot.xticks([1, 5, 10, 15, 20])
pyplot.title('Spectrogram Correlation')
pyplot.xlabel('Trial')
pyplot.ylabel('Pearson Corr.')
pyplot.grid()
pyplot.ylim(0.7, 1.025)
pyplot.tight_layout()
image_output = path.join(Settings.image_folder, 'strength.pdf')
pyplot.savefig(image_output)
pyplot.show()
