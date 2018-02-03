# -*- coding: utf-8 -*-
import numpy
from matplotlib import pyplot


def lif(v, ge, gi, i):
    dv = (v * -0.01) + ge - gi + i
    spk = v > 1
    dv[spk] = -v[spk]
    return dv, spk

def lif_net(num_neurons, duration):
    offset = -numpy.linspace(0, 4 * numpy.pi, num_neurons)
    offset[:num_neurons / 2] = -3 * numpy.pi
    v = numpy.zeros((duration, num_neurons))
    ge = numpy.zeros(num_neurons)
    gi = numpy.zeros(num_neurons)
    i = 0.019 * numpy.random.rand(duration, num_neurons)
    spikes = numpy.zeros((duration, num_neurons))
    v[0,:] = numpy.random.rand(num_neurons)
    for t in numpy.arange(1, duration):
        ge[num_neurons / 2:] = 0.15 * spikes[t-1,:num_neurons / 2]
        gi = numpy.ones(num_neurons) * 0.001 * (numpy.sin(offset + t / 100) + 1)
        dv, spikes[t,:] = lif(v[t-1,:], ge, gi, i[t,:])
        v[t,:] = v[t-1,:] + dv
    return spikes

spikes = lif_net(2000, 3000)
indices = numpy.where(spikes)
pyplot.figure()
ax = pyplot.subplot(121)
pyplot.scatter(indices[0][indices[1] < 1000], indices[1][indices[1] < 1000], marker='.', alpha=0.5)

indices = numpy.where(spikes)
pyplot.scatter(indices[0][indices[1] >= 1000], indices[1][indices[1] >= 1000], marker='.', alpha=0.5)

pyplot.xlabel('Time (ms)')
pyplot.yticks([])

pyplot.subplot(164)
pyplot.hist(indices[1], bins=50, orientation='horizontal')
pyplot.yticks([])
pyplot.xticks([])

pyplot.tight_layout()
