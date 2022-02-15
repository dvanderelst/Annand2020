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

pyplot.close('all')

sets = [data1, data5, data10, data15, data20, ind, dyad]
xcorrs = []
rmss = []

for nr, data in enumerate(sets):
    x = data.x
    y = data.y
    dx = data.dx
    dy = data.dy

    dx = dx - numpy.mean(dx)
    dy = dy - numpy.mean(dy)
    dx = dx / numpy.max(numpy.abs(dx))
    dy = dy / numpy.max(numpy.abs(dy))

    radians_x = numpy.arccos(dx)
    derivative = numpy.gradient(dx)
    radians_x[derivative > 0] = - radians_x[derivative > 0]

    radians_y = numpy.arccos(dy)
    derivative = numpy.gradient(dy)
    radians_y[derivative > 0] = - radians_y[derivative > 0]

    xcor = CrossCorr.short_term_xcorr(radians_x, radians_y)
    rms = CrossCorr.short_term_rms(data.dx, data.dy)

    xcorrs.append(xcor)
    rmss.append(rms)

    #pyplot.subplot(3,3,nr+1)
    #pyplot.hist(degrees)
    difference = radians_x - radians_y
    degrees = numpy.degrees(difference)
    degrees = degrees % 360

    pyplot.subplot(3,3,nr+1)
    pyplot.hist(degrees)


pyplot.show()

pyplot.figure()
pyplot.subplot(2,1,1)
pyplot.plot(rmss[:-2])
pyplot.subplot(2,1,2)
pyplot.plot(xcorrs[:-2])
pyplot.show()
