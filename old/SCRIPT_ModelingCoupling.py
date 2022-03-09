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


def generate_noise(cor, shift):
    corr_mat = numpy.array([[1, cor], [cor, 1]])
    diag = numpy.diag([1, 1])

    cov_mat = numpy.matmul(diag , diag)
    cov_mat = numpy.matmul(cov_mat , corr_mat)

    noise = multivariate_normal([0, 0], cov=cov_mat, size=600)
    dx = noise[:, 0]
    dy = noise[:, 1]

    dx = Misc.smooth_signal(dx, 7, window='box')
    dy = Misc.smooth_signal(dy, 7, window='box')

    var_dx = numpy.var(dx)
    var_dy = numpy.var(dy)

    dx = dx / var_dx**0.5
    dy = dy / var_dy**0.5

    dy = numpy.roll(dy, shift)
    return dx, dy

#%%
noise = numpy.random.normal(size=600)
noise = Misc.smooth_signal(noise, 15, window='box')

x1 = CrossCorr.cross_correlate(data1.dx, data1.dx)
x20 = CrossCorr.cross_correlate(data20.dx, data20.dx)
xn = CrossCorr.cross_correlate(noise, noise)
pyplot.plot(x1)
pyplot.plot(x20)
pyplot.plot(xn)
pyplot.show()

#%%
repeats = 250
n_correlations = 15

correlations = numpy.linspace(0, 1, n_correlations)
shifts = numpy.array([0, 1, 5, 15])
n_shifts = len(shifts)

results = numpy.zeros((n_correlations, n_shifts, repeats))

for r in range(repeats):
    for i, c in enumerate(correlations):
        for j, s in enumerate(shifts):
            dx, dy = generate_noise(c, s)
            error = numpy.sqrt(dx ** 2 + dy ** 2)
            error = numpy.mean(error)
            results[i,j,r] = error

results = numpy.mean(results, axis=2)
results = 100 *(results / numpy.max(results))
pyplot.plot(correlations, results)
pyplot.legend(['$\delta t = 0$', '$\delta t = 0.1$','$\delta t = 0.5$', '$\delta t = 1.5$'])
pyplot.ylabel('RMSE (normalized, $\%$)')
pyplot.xlabel('Coupling strength (error correlation, $r$)')
pyplot.tight_layout()
output_file = 'latex/correlation.pdf'
pyplot.savefig(output_file)
pyplot.show()

