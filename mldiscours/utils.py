from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class LivePlotHistory(Callback):
    def __init__(self):
        self.batch_log_values = []
        self.markers = []
        self.fig, self.axes = plt.subplots()

    def animate(self, x, y):
        self.lines[0].set_data(x, y)
        return self.lines

    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']
        self.nb_epoch = self.params['nb_epoch']
        self.nb_sample = self.params['nb_sample']
        self.X = np.zeros(self.nb_sample)
        self.Y = np.zeros(self.nb_sample)
        self.lines = [self.axes.plot(self.X, self.Y, animated=True)[0]]
        self.anim = FuncAnimation(self.fig, self.animate(self.X, self.Y), interval=5, blit=True)
        plt.show()


    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        self.b_seen = 0
        self.b_seen_index = 0

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.b_seen += batch_size
        self.b_seen_index +=1

        # for k in self.params['metrics']:
        #     if k in logs:
        #         self.batch_log_values.append((k, logs[k]))

        if self.verbose and self.b_seen < self.params['nb_sample']:
            self.X[self.b_seen_index] = self.b_seen_index
            self.Y[self.b_seen_index] = logs['loss']
            self.animate(self.X, self.Y)


    def on_epoch_end(self, epoch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))


def test_live_plot():
    X = np.linspace(0, 2, 1000)
    Y = X ** 2 + np.random.random(X.shape)
    plt.ion()
    graph = plt.plot(X, Y)[0]
    while True:
        Y = X ** 2 + np.random.random(X.shape)
        graph.set_ydata(Y)
        plt.pause(0.0001)


def test_live_plot2():
    X = np.zeros(1000)
    Y = X ** 2 + np.random.random(X.shape)
    plt.ion()
    figure = plt.figure()
    ax = figure.add_subplot(111)
    for i in range(X.shape[0]):
        X[i] = i
        Y = X ** 2 + np.random.random(X.shape)
        ax.plot(X[:i], Y[:i], '-rD')
        plt.draw()
        plt.pause(0.0001)



import matplotlib.animation as animation

class DrawWeights(Callback):

    def __init__(self, figsize, layer_id=0, param_id=0):
        self.layer_id = layer_id
        self.param_id = param_id
        # Initialize the figure and axis
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(1, 1, 1)

    def on_train_begin(self, logs={}):
        self.imgs = []
        self.verbose = self.params['verbose']
        self.nb_epoch = self.params['nb_epoch']
        self.nb_sample = self.params['nb_sample']
        self.X = np.zeros(self.nb_sample)
        self.Y = np.zeros(self.nb_sample)

    def on_epoch_begin(self, epoch, logs={}):
        self.b_seen = 0
        self.b_seen_index = 0

    def on_batch_end(self, batch, logs={}):
        if batch % 5 == 0:
            # Access the full weight matrix
            batch_size = logs.get('size', 0)
            self.b_seen += batch_size
            self.b_seen_index += 1

            self.X[self.b_seen_index] = self.b_seen_index
            self.Y[self.b_seen_index] = logs['loss']
            # Create the frame and add it to the animation
            img = self.ax.plot(self.X, self.Y)
            self.imgs.append(img)

    def on_train_end(self):
        # Once the training has ended, display the animation
        anim = animation.ArtistAnimation(self.fig, self.imgs, interval=10, blit=False)
        plt.show()

