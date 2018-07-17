# -*- coding: utf-8 -*-
"""
Applies ResNet50 network to large images dynamically to make decisions about which objects are present.
"""
from threading import Thread
from keras.preprocessing import image as image_utils
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


def slice_subimage(image, offset, dimension=(224,224)):
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
plot_image(ax, slice_subimage(data, (0.5, 0.5)))
pyplot.tight_layout()

#%%

model = ResNet50(weights="imagenet")

predictions = model.predict(slice_subimage(data, (0.5, 0.5)))
decoded = decode_predictions(predictions)
for i in range(5):
    print('{} {:.0%}'.format(decoded[0][i][1], decoded[0][i][2]))

pyplot.plot(predictions.transpose())

#%%


class Region:
    def __init__(self, offset, bounds, dimension=(224, 224)):
        self.offset = offset
        self.bounds = bounds
        self.dimension = dimension


class SubimageClassifier:
    def __init__(self, model):
        self.model = model
        self.original, self.data, self.display = load_image('./jake.jpg')
        self.region = Region((0.5, 0.5), (self.data.shape[0], self.data.shape[1]))
        self.subimage = slice_subimage(self.data, self.region.offset)
        self.predictions = numpy.zeros(1000)
    
    def load_image(self, filename, scale=1.0):
        try:
            self.original, self.data, self.display = load_image(filename, scale)
        except:
            return False
        return self.data.shape[0] > self.region.dimension[0] and self.data.shape[1] > self.region.dimension[1]
    
    def classify_subimage(self):
        self.region.offset = numpy.random.rand(2)
        self.subimage = slice_subimage(self.data, self.region.offset)
        self.predictions = self.model.predict(self.subimage)


def show_running_predictions(ax, preds):
    ax.imshow(preds.transpose(), aspect='auto', vmin=0, vmax=1)


def plot_top_prediction_classes(ax, preds):
    max_preds = preds.max(axis=0).reshape((1, 1000))
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


def update_classifier(classifier, predictions, row):
    classifier.classify_subimage()
    predictions[row,:] = classifier.predictions


def click_figure(event):
    if event.button == 1 and event.inaxes == event.canvas.active_axes:
        region = event.canvas.region
        region.offset = (event.ydata / region.bounds[0], event.xdata / region.bounds[1])


def close_figure(event):
    event.canvas.closed = True


#%%

imageset = pandas.read_csv('C:/Users/bimmy/git/code-everyday/open_images_urls.csv', delimiter=',')

#%%

# Initialize the classifier and timeseries
classifier = SubimageClassifier(model)
classifier.load_image(imageset.image_url[67], 0.5)
preds = numpy.zeros((100, 1000))
conf = numpy.zeros(1000)
conf2 = numpy.zeros(1000)
t = 0
rt = 0

# Initialize the classifer subplots and event handlers
fig = pyplot.figure(figsize=(12,12))
fig.canvas.closed = False
fig.canvas.region = classifier.region
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

# Initialize the classifer update thread
update_thread = Thread(target=update_classifier, args=(classifier, preds, t % 100))
update_thread.start()

# Run this loop until the user exits the interactive plot
while not fig.canvas.closed:
    # While waiting for the next predictions, handle plot events
    if update_thread.is_alive():
        pyplot.pause(0.001)
    # Update the display with the next set of predictions
    else:
        update_thread.join()
        # Show a running map of prediction confidence for all 1000 classes
        ax1.clear()
        show_running_predictions(ax1, preds)
        # Show the timeseries for the top 5 predictions
        ax2.clear()
        plot_top_prediction_classes(ax2, preds)
        # Show the current subimage
        ax3.clear()
        plot_image(ax3, classifier.subimage)
        # Show the original image and the subimage boundary
        ax4.clear()
        ax4.imshow(classifier.display)
        draw_subimage_border(ax4, classifier.display, classifier.region.offset)
        # Calculate a running confidence estimate for each class and plot the values
        conf = 0.99 * conf + 0.01 * preds[t % 100,:]
        conf2 = 0.99 * conf2 + 0.01 * preds[t % 100,:]
        ax5.clear()
        ax5.plot(conf2)
        # Print the highest confidence class when the running confidence estimate exceeds 0.25
        if conf2.max() > 0.25 and rt == 0:
            label = decode_predictions(numpy.expand_dims(conf2, axis=0), top=1)[0][0][1]
            rt = t
            pyplot.suptitle('Label: {} ({} steps)'.format(label, rt))
        # Show the updated plots
        pyplot.show()
        # Start the update thread for the next set of predictions
        t += 1
        update_thread = Thread(target=update_classifier, args=(classifier, preds, t % 100))
        update_thread.start()


#%%

# Initialize the classifier and timeseries
classifier = SubimageClassifier(model)
image_num = numpy.random.randint(100000)
classifier.load_image(imageset.image_url[image_num], 0.5)
preds = numpy.zeros((1, 1000))
conf = numpy.zeros(1000)
conf2 = numpy.zeros(1000)
t = 0
decision = False
images = []
labels = []
dt = []

# Initialize the classifer subplots and event handlers
fig = pyplot.figure(figsize=(12,12))
fig.canvas.closed = False
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)
fig.canvas.mpl_connect('close_event', close_figure)
fig.show()
fig.canvas.draw()

# Initialize the classifer update thread
update_thread = Thread(target=update_classifier, args=(classifier, preds, 0))
update_thread.start()

# Classify 200 images or until the user closes the interactive plot
while (not fig.canvas.closed) and (len(dt) < 1):
    # If the classifier has made a decision for the current image, select the next image
    if decision:
        conf.fill(0)
        conf2.fill(0)
        t = 0
        ax1.set_title('')
        image_num = numpy.random.randint(100000)
        if classifier.load_image(imageset.image_url[image_num], 0.5):
            decision = False
        update_thread = Thread(target=update_classifier, args=(classifier, preds, 0))
        update_thread.start()
    # While waiting for the next predictions, handle plot events
    elif update_thread.is_alive():
        pyplot.pause(0.001)
    # Update the display with the next set of predictions
    else:
        update_thread.join()
        # Show a running map of prediction confidence for all 1000 classes
        ax1.clear()
        ax1.imshow(classifier.display)
        draw_subimage_border(ax1, classifier.display, classifier.region.offset)
        # Calculate a running confidence estimate for each class and plot the values
        conf = 0.99 * conf + 0.01 * preds[0,:]
        conf2 = 0.99 * conf2 + 0.01 * preds[0,:]
        ax2.clear()
        ax2.plot(conf2)
        ax2.plot([0, 1000], [0.125, 0.125], 'g--')
        ax2.set_ylim(0, 0.15)
        # Print the highest confidence class when the running confidence estimate exceeds 0.25
        if conf2.max() > 0.125 or t == 150:
            # Record the trial data
            label = decode_predictions(numpy.expand_dims(conf2, axis=0), top=1)[0][0][1]
            labels.append(label)
            images.append(image_num)
            dt.append(t)
            decision = True
            # Display the result
            ax1.set_title('Label: {} ({} steps)'.format(label, t))
            ax3.clear()
            ax3.plot(dt, 'o')
            pyplot.show()
            pyplot.pause(0.25)
        else:
            # Show the updated plots
            pyplot.show()
            # Start the update thread for the next set of predictions
            t += 1
            update_thread = Thread(target=update_classifier, args=(classifier, preds, 0))
            update_thread.start()

#%%

results = pandas.DataFrame({
        'image_name': imageset.image_name[images].values,
        'label':labels,
        'time':dt})
results.to_csv('results3.csv', index=False)

#%%


from scipy.stats import truncexpon

x = numpy.linspace(0, 150, 1000)
pyplot.plot(x, truncexpon.cdf(x, 150, loc=2.7, scale=52.7))
pyplot.plot(numpy.sort(results.time.values), numpy.linspace(0, 1, 1000))

#results = pandas.read_csv('results.csv')
#b = 149
#y = results.time.values[results.time.values < 150]
#time_dist = truncexpon.fit(y, b)
#x = numpy.linspace(truncexpon.ppf(0.01, b), truncexpon.ppf(0.99, b), 100)
#x = numpy.linspace(0, 150, 100)
#pyplot.plot(x, time_dist.pdf(x, b))

#pyplot.hist(results.time.values[results.time.values < 150], bins=50, normed=True)
#pyplot.plot(numpy.linspace(1, 150, 1000), 1 - numpy.exp(-(1 / 50) * numpy.linspace(1, 150, 1000)))
#pyplot.plot(numpy.sort(results.time.values)[1:], numpy.linspace(0, 1, 999) / numpy.diff(numpy.sort(results.time.values)))
