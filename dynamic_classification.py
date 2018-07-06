# -*- coding: utf-8 -*-
"""
Applies VGG16 network to large images dynamically to make decisions about which objects are present.
"""
from threading import Thread
from keras.preprocessing import image as image_utils
#from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from matplotlib import pyplot
from scipy.misc import imresize
from skimage import io
import numpy
import cv2
import pandas

#%%

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%

def load_image(filename, scale=1.0):
    original = io.imread(filename)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = imresize(original, scale)
    data = image_utils.img_to_array(original)
    display = preprocess_input(original).squeeze()
    display -= display.min()
    display /= display.max()
    return original, data, display


def subimage(image, offset, dimension=(224,224)):
    x = int(offset[1] * (image.shape[1] - dimension[1]))
    y = int(offset[0] * (image.shape[0] - dimension[0]))
    sub = image[y:y + dimension[0], x:x + dimension[1], :].copy()
    sub = numpy.expand_dims(sub, axis=0)
    sub = preprocess_input(sub)
    return sub


def plot_image(ax, image):
    scaled_image = image.copy().squeeze()
    scaled_image -= scaled_image.min()
    scaled_image /= scaled_image.max()
    ax.imshow(scaled_image)


#%%

original, data, display = load_image('C:/Users/bimmy/Pictures/London.jpg', 0.75)
pyplot.figure(figsize=(12,6))
ax = pyplot.subplot(121)
plot_image(ax, display)
ax = pyplot.subplot(122)
plot_image(ax, subimage(data, (0.5, 0.5)))
pyplot.tight_layout()

#%%

model = ResNet50(weights="imagenet")

predictions = model.predict(subimage(data, (0.5, 0.5)))
decoded = decode_predictions(predictions)
for i in range(5):
    print('{} {:.0%}'.format(decoded[0][i][1], decoded[0][i][2]))

pyplot.plot(predictions.transpose())

#%%

def show_running_predictions(ax, preds):
    p = preds.copy()
    ax.imshow(p.transpose(), aspect='auto')


def plot_top_prediction_classes(ax, preds):
    p = preds.copy()
    max_preds = p.max(axis=0).reshape((1, 1000))
    order = numpy.argsort(max_preds)
    decoded = decode_predictions(max_preds)
    for i in range(5):
        ax.plot(preds[:,order[0][-1 - i]].transpose(), label=decoded[0][i][1])
    ax.legend()


def draw_subimage_border(ax, image, offset, dimension=(224,224)):
    x0 = int(offset[1] * (image.shape[1] - dimension[1]))
    y0 = int(offset[0] * (image.shape[0] - dimension[0]))
    x1 = x0 + dimension[1]
    y1 = y0 + dimension[0]
    ax.plot([x0, x0, x1, x1, x0], [y0, y1, y1, y0, y0], 'g--')


class Region:
    def __init__(self, offset, bounds, dimension=(224, 224)):
        self.offset = offset
        self.bounds = bounds
        self.dimension = dimension


def update_model(model, image, preds, t, region):
    region.offset = numpy.clip(numpy.random.rand(2), 0, 1)
    sub = subimage(image, region.offset)
    preds[t,:] = model.predict(sub)


def click_figure(event):
    if event.button == 1 and event.inaxes == event.canvas.active_axes:
        region = event.canvas.region
        region.offset = (event.ydata / region.bounds[0], event.xdata / region.bounds[1])


def close_figure(event):
    event.canvas.closed = True


#%%

imageset = pandas.read_csv('C:/Users/bimmy/git/code-everyday/open_images_dataset_urls.csv', delimiter=',')

#%%

original, data, display = load_image(imageset.image_url[63], 1.0)
region = Region((0.5, 0.5), (data.shape[0], data.shape[1]))
t = 0
preds = numpy.zeros((100, 1000))
conf = numpy.zeros(1000)
conf2 = numpy.zeros(1000)
decisions = []

fig = pyplot.figure(figsize=(12,12))
fig.canvas.closed = False
fig.canvas.region = region
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(223)
ax5 = fig.add_subplot(224)
fig.canvas.active_axes = ax4
fig.canvas.mpl_connect('button_press_event', click_figure)
fig.canvas.mpl_connect('close_event', close_figure)
fig.show()
fig.canvas.draw()
model_thread = Thread(target=update_model, args=(model, data, preds, t, region))
model_thread.start()

while not fig.canvas.closed:
    if model_thread.is_alive():
        pyplot.pause(0.001)
    else:
        model_thread.join()
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()
        show_running_predictions(ax1, preds)
        plot_top_prediction_classes(ax2, preds)
        plot_image(ax3, subimage(data, region.offset))
        ax4.imshow(display)
        draw_subimage_border(ax4, display, region.offset)
        conf = 0.99 * conf + 0.01 * preds[i,:]
        conf2 = 0.99 * conf2 + 0.01 * preds[i,:]
        ax5.plot(conf2)
        if conf2.max() > 0.25:
            label = decode_predictions(numpy.expand_dims(conf2, axis=0), top=1)[0][0][1]
            print('t: {} label: {}'.format(t, label))
        pyplot.show()
        t += 1
        i = t % 100
        model_thread = Thread(target=update_model, args=(model, data, preds, i, region))
        model_thread.start()
