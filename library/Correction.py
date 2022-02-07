import numpy

def run_corrections(targets, predicted_samples, sd=0, p=0.9):
    n = targets.shape[0]
    traces = []
    for i in range(n):
        trace0, trace1 = run_correction(targets, i, predicted_samples, sd, p)
        trace =trace0 + trace1
        traces.append(trace)
    traces = numpy.array(traces)
    return traces

def run_correction(targets, example, predicted_samples, sd=0, p=0.9):
    out0 = targets[example, 0:predicted_samples]
    out1 = targets[example, predicted_samples:]

    position0 = 0
    trace0 = [position0]

    position1 = 0
    trace1 = [position1]

    for i in range(len(out0) - 1):
        next_target0 = out0[i + 1]
        error0 = next_target0 - position0
        random_error0 = numpy.random.normal(0, sd)
        position0 = position0 + (error0 * p) + random_error0
        trace0.append(position0)

        next_target1 = out1[i + 1]
        error1 = next_target1 - position1
        random_error1 = numpy.random.normal(0, sd)
        position1 = position1 + (error1 * p) + random_error1
        trace1.append(position1)
    return trace0, trace1
