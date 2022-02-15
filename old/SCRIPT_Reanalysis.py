import numpy
from matplotlib.lines import Line2D
from library import Misc
from library import CrossCorr
from matplotlib import pyplot

data1 = Misc.read_csv('digitize/trial1.csv', 0, 60)
data5 = Misc.read_csv('digitize/trial5.csv', 0, 60)
data10 = Misc.read_csv('digitize/trial10.csv', 0, 60)
data15 = Misc.read_csv('digitize/trial15.csv', 0, 60)
data20 = Misc.read_csv('digitize/trial20.csv', 0, 60)

ind = Misc.read_csv('digitize/ind.csv', 20, 40)
dyad = Misc.read_csv('digitize/dyad.csv', 10, 50)
# %% Get overall delays
_, _, _, tr1 = Misc.lag_finder(data1.dx, data1.dy, plot=True)
_, _, _, tr5 = Misc.lag_finder(data5.dx, data5.dy, plot=True)
_, _, _, tr10 = Misc.lag_finder(data10.dx, data10.dy, plot=True)
_, _, _, tr15 = Misc.lag_finder(data15.dx, data15.dy, plot=True)
_, _, _, tr20 = Misc.lag_finder(data20.dx, data20.dy, plot=True)

traces = [tr1, tr5, tr10, tr15, tr20]
traces = numpy.array(traces)
traces = numpy.transpose(traces)
# %% Get Short term xcorrs
xcor1 = CrossCorr.short_term_xcorr(data1.radians_x, data1.radians_y)
xcor5 = CrossCorr.short_term_xcorr(data5.radians_x, data5.radians_y)
xcor10 = CrossCorr.short_term_xcorr(data10.radians_x, data10.radians_y)
xcor15 = CrossCorr.short_term_xcorr(data15.radians_x, data15.radians_y)
xcor20 = CrossCorr.short_term_xcorr(data20.radians_x, data20.radians_y)
xcorrs = [xcor1, xcor5, xcor10, xcor15, xcor20]

xcor_ind = CrossCorr.short_term_xcorr(ind.radians_x, ind.radians_y)
xcor_dyad = CrossCorr.short_term_xcorr(dyad.radians_x, dyad.radians_y)

# %% Get Short term rms
rms1 = CrossCorr.short_term_rms(data1.dx, data1.dy)
rms5 = CrossCorr.short_term_rms(data5.dx, data5.dy)
rms10 = CrossCorr.short_term_rms(data10.dx, data10.dy)
rms15 = CrossCorr.short_term_rms(data15.dx, data15.dy)
rms20 = CrossCorr.short_term_rms(data20.dx, data20.dy)
rmss = [rms1, rms5, rms10, rms15, rms20]

rms_ind = CrossCorr.short_term_rms(ind.dx, ind.dy)
rms_dyad = CrossCorr.short_term_rms(dyad.dx, dyad.dy)
# %%
line1 = Line2D([0], [0], label='Analysis fig. 5', color='k', marker='s')
line2 = Line2D([0], [0], label='Analysis fig. 7, ind.', color='k', marker='D', linestyle='none')
line3 = Line2D([0], [0], label='Analysis fig. 7, dyad', color='k', marker='X', linestyle='none')
handles = [line1, line2, line3]

test = [rms_ind, xcor_ind, rms_dyad, xcor_dyad]

trial_numbers = [1, 5, 10, 15, 20]

pyplot.figure(figsize=(5, 3))
pyplot.plot(trial_numbers, xcorrs, 'r', marker='s')
pyplot.scatter(20.5, xcor_dyad, c='r', marker='D')
pyplot.scatter(20.5, xcor_ind, c='r', marker='X')

pyplot.ylabel('Average short term xcorr', color='r')

ax = pyplot.twinx()
ax.plot(trial_numbers, rmss, 'b', marker='s')
ax.scatter(21, rms_dyad, c='b',  marker='D')
ax.scatter(21, rms_ind, c='b',  marker='X')

ax.set_ylabel('Average short term RMSE', color='b')
ax.set_xticks(trial_numbers)
ax.set_xlabel('Trial Number')
pyplot.tight_layout()

pyplot.legend(handles=handles)

output_file = 'latex/reanalysis.pdf'
pyplot.savefig(output_file)

pyplot.show()

#%%
pyplot.plot(data1.dx, data1.dy)
pyplot.show()

pyplot.plot(data10.dx, data10.dy)
pyplot.show()

pyplot.plot(data20.dx, data20.dy)
pyplot.show()