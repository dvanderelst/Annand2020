import numpy

import Misc
import CrossCorr
from matplotlib import pyplot
from numpy.random import  multivariate_normal

pyplot.rcParams['text.usetex'] = True

data1 = Misc.read_csv('digitize/trial1.csv', 0, 60)
data5 = Misc.read_csv('digitize/trial5.csv', 0, 60)
data10 = Misc.read_csv('digitize/trial10.csv', 0, 60)
data15 = Misc.read_csv('digitize/trial15.csv', 0, 60)
data20 = Misc.read_csv('digitize/trial20.csv', 0, 60)

pyplot.hist(data20.dx)
pyplot.hist(data20.dy)
pyplot.show()



#%%

set = data20
errors = numpy.concatenate((set.dx.values, set.dy.values))
std = numpy.std(errors)
std = 1
correlations = numpy.linspace(0, 1, 10)
shifts = numpy.arange(0, 2)

results = numpy.zeros((10, 2))


for i, c in enumerate(correlations):
    for j, s in enumerate(shifts):

        corr_mat = numpy.array([[1, c], [c, 1]])
        diag = numpy.diag([std, std])

        cov_mat = numpy.matmul(diag , diag)
        cov_mat = numpy.matmul(cov_mat , corr_mat)

        noise = multivariate_normal([0, 0], cov=cov_mat, size=300)

        noise1 = noise[:,0]
        noise2 = noise[:, 1]

        noise1 = Misc.smooth_signal(noise1, 4)
        noise2 = Misc.smooth_signal(noise2, 4)
        noise[:, 0] = noise1
        noise[:, 1] = noise2

        noise1 = numpy.roll(noise1, s)
        noise[:, 0] = noise1

        error = numpy.sum(noise ** 2, axis=1)
        error = numpy.sqrt(error)
        error = numpy.mean(error)

        t1 = numpy.corrcoef(numpy.transpose(noise))
        t2 = numpy.cov(numpy.transpose(noise))

        results[i,j] = error


Misc.lag_finder(noise1, noise2, 0.1, plot=True)
Misc.lag_finder(set.dx, set.dy, 0.1, plot=True)

#%%
noise1 = noise[:,0]
sm = Misc.smooth_signal(noise1,4)

x1 = CrossCorr.cross_correlate(set.dx, set.dx)
x2 = CrossCorr.cross_correlate(sm, sm)

pyplot.subplot(2,1,1)
pyplot.plot(x1)
pyplot.subplot(2,1,2)
pyplot.plot(x2)
pyplot.show()


# #%%
# n = 100*(results/numpy.max(results))
#
# pyplot.figure(figsize=(4,3))
# pyplot.plot(correlations, n)
# pyplot.ylabel('RMSE (normalized, $\%$)')
# pyplot.xlabel('Coupling strength (error correlation, $r$)')
# pyplot.legend(['$\delta t = 0\ samples$', '$\delta t = 1\ sample$'])
#
# pyplot.tight_layout()
# output_file = 'latex/correlation.pdf'
# pyplot.savefig(output_file)
# pyplot.show()