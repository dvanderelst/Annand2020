import Misc
import cross
from matplotlib import pyplot


data1 = Misc.read_csv('digitize/trial1.csv')
data5 = Misc.read_csv('digitize/trial5.csv')
data10 = Misc.read_csv('digitize/trial10.csv')
data15 = Misc.read_csv('digitize/trial15.csv')
data20 = Misc.read_csv('digitize/trial20.csv')

Misc.plot_data(data1)

pyplot.plot(data1.t, data1.dx)
pyplot.plot(data1.t, data1.dy)
pyplot.show()

pyplot.plot(data5.t, data5.dx)
pyplot.plot(data5.t, data5.dy)
pyplot.show()

pyplot.plot(data10.t, data10.dx)
pyplot.plot(data10.t, data10.dy)
pyplot.show()

pyplot.plot(data15.t, data15.dx)
pyplot.plot(data15.t, data15.dy)
pyplot.show()

pyplot.plot(data20.t, data20.dx)
pyplot.plot(data20.t, data20.dy)
pyplot.show()

# %%
corr1, rms1, delay1, tr1 = Misc.lag_finder(data1.dx, data1.dy, plot=True)
corr5, rms5, delay5, tr5 = Misc.lag_finder(data5.dx, data5.dy, plot=True)
corr10, rms10, delay10, tr10 = Misc.lag_finder(data10.dx, data10.dy, plot=True)
corr15, rms15, delay15, tr15 = Misc.lag_finder(data15.dx, data15.dy, plot=True)
corr20, rms20, delay20, tr20 = Misc.lag_finder(data20.dx, data20.dy, plot=True)

# %%
lags = [delay1, delay5, delay10, delay15, delay20]
corrs = [corr1, corr5, corr10, corr15, corr20]
rms = [rms1, rms5, rms10, rms15, rms20]
pyplot.plot(corrs, 'r')
ax = pyplot.twinx()
ax.plot(rms, 'b')
pyplot.show()

pyplot.plot(rms,corrs)
pyplot.show()

#%%
pyplot.plot(tr1)
pyplot.plot(tr5)
pyplot.plot(tr10)
pyplot.plot(tr15)
pyplot.plot(tr20)
pyplot.legend(['Trial 1', 'Trial5', 'Trial10', 'Trial15', 'Trial20'])
pyplot.show()

#%%
a = cross.short_term_xcorr(data1.dx, data1.dy)
b = cross.short_term_xcorr(data5.dx, data5.dy)
c = cross.short_term_xcorr(data10.dx, data10.dy)
d = cross.short_term_xcorr(data15.dx, data15.dy)
e = cross.short_term_xcorr(data20.dx, data20.dy)

corrs = [a,b,c,d,e]

pyplot.plot(corrs, 'r')
pyplot.ylabel('Average short term xcorr', color='r')
ax = pyplot.twinx()
ax.plot(rms, 'b')
ax.set_ylabel('RMS error', color='b')
pyplot.show()