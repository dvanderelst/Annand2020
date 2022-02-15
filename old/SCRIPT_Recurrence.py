import numpy
from matplotlib import pyplot
from pyrqa.analysis_type import Cross
from pyrqa.computation import RQAComputation
from pyrqa.computation import RPComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.neighbourhood import Unthresholded

from library import Misc

data1 = Misc.read_csv('digitize/trial1.csv', 0, 60)
data5 = Misc.read_csv('digitize/trial5.csv', 0, 60)
data10 = Misc.read_csv('digitize/trial10.csv', 0, 60)
data15 = Misc.read_csv('digitize/trial15.csv', 0, 60)
data20 = Misc.read_csv('digitize/trial20.csv', 0, 60)

ind = Misc.read_csv('digitize/ind.csv', 20, 40)
dyad = Misc.read_csv('digitize/dyad.csv', 10, 50)

sets = [data1, data5, data10, data15, data20, ind, dyad]
setnames = ['data1', 'data5', 'data10', 'data15', 'data20', 'ind', 'dyad']
indicators = []

for name, data_to_use in zip(setnames, sets):

    max_x = numpy.max(numpy.abs(data_to_use.dx))
    max_y = numpy.max(numpy.abs(data_to_use.dy))

    x = data_to_use.dx / max_x
    y = data_to_use.dx / max_y

    time_series_x = TimeSeries(x, embedding_dimension=10)
    time_series_y = TimeSeries(y, embedding_dimension=10)
    time_series = (time_series_x, time_series_y)

    settings = Settings(time_series,
                        analysis_type=Cross,
                        neighbourhood=FixedRadius(0.7),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=0)

    # analysis
    computation = RQAComputation.create(settings)
    analytics = computation.run()
    indicators.append(analytics.determinism)
    print(analytics)

    # plot
    computation = RPComputation.create(settings)
    matrix = computation.run()

    pyplot.imshow(matrix.recurrence_matrix, vmin=0, vmax=1)
    pyplot.colorbar()
    pyplot.title(name)
    pyplot.savefig(name + '.png')
    pyplot.show()


# ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
#                                     'cross_recurrence_plot.png')
pyplot.plot(indicators[0:5])
pyplot.show()