# -*- coding: utf-8 -*-
"""
This script explores generation of RT-like distributions using a simulated serial
evaluation process.
"""
import numpy
from matplotlib import pyplot
from scipy.signal import savgol_filter


#%% This is some kind of lame DDM stuff

def generate_signals(in1=1.01, in2=1.0, duration=1000, rate=10, frequency=10):
    t = numpy.linspace(0, duration / 1000, duration)
    n1 = numpy.random.poisson(rate, duration) + 1
    s1 = numpy.random.poisson(in1 * rate, duration)
    s2 = numpy.random.poisson(in2 * rate, duration)
    oscillation = rate * (numpy.sin(2 * numpy.pi * frequency * t) + 1)
    n2 = numpy.random.poisson(oscillation, duration) + 1
    s3 = numpy.random.poisson(in1 * oscillation, duration)
    s4 = numpy.random.poisson(in2 * oscillation, duration)
    return t, n1, s1, s2, n2, s3, s4

t, n1, s1, s2, n2, s3, s4 = generate_signals()

def plot_smooth(x, y, label):
    pyplot.plot(x, savgol_filter(y, 21, 3), lw=1.25, label=label)

pyplot.figure()
ax = pyplot.subplot(221)
plot_smooth(t, s1, 'Steady Signal')
plot_smooth(t, s3, 'Oscillating Signal')
pyplot.xlim(0, 1)
#pyplot.ylim(0, 3)
pyplot.ylabel('Activity')
ax.set_xticklabels([])
pyplot.yticks([])
pyplot.legend()

pyplot.subplot(223)
pyplot.plot(t, numpy.cumsum(s1 - s2), lw=1.25)
pyplot.plot(t, numpy.cumsum(s3 - s4), lw=1.25)
pyplot.xlim(0, 1)
pyplot.xlabel('Time (s)')
pyplot.ylabel('Activity - Noise')
pyplot.yticks([])

max_s1_e = numpy.zeros(1000)
max_s3_e = numpy.zeros(1000)
for i in range(1000):
    t, n1, s1, s2, n2, s3, s4 = generate_signals(duration=10000)
    max_s1_e[i] = numpy.cumsum(s1 - (n1 - 1)).max()
    max_s3_e[i] = numpy.cumsum(s3 - (n2 - 1)).max()

pyplot.subplot(222)
pyplot.plot(numpy.sort(max_s1_e), numpy.linspace(0, 1, 1000), lw=1.5)
pyplot.plot(numpy.sort(max_s3_e), numpy.linspace(0, 1, 1000), lw=1.5)
pyplot.xlabel('Maximum Activity - Noise')
pyplot.ylabel('Cum. Prob. of (x)')

#%% Dynamic decision making


xMax = 100
tMax = 100
img = numpy.random.randn(xMax, xMax)
sample = numpy.zeros(tMax)
dx = 5 * numpy.random.randn(xMax, 2)
x = numpy.zeros((tMax, 2))
x[0,:] = (xMax / 2, xMax / 2) + dx[0,:]
for t in range(1, len(dx)):
    x[t,:] = numpy.maximum(numpy.minimum(x[t-1] + dx[t], xMax - 0.5), 0.5)
    sample[t] = img[int(x[t,0]), int(x[t,1])]

pyplot.subplot(211)
pyplot.imshow(img)
pyplot.plot(x[:,0], x[:,1], 'r-')
pyplot.plot(x[-1,0], x[-1,1], 'k+', markersize=15)
pyplot.subplot(212)
pyplot.plot(sample, '.')
