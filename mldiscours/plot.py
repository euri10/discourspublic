import math
import numpy as np
import matplotlib
import time

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import Callback
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


class Scope(object):
    def __init__(self, ax):
        self.ax = ax
        self.x = [0]
        self.y = [0]
        self.line = Line2D(self.x, self.y)
        self.ax.add_line(self.line)
        self.ax.set_ylim(-.1, 1.1)
        self.ax.set_xlim(0, len(self.x))
        plt.ion()

    def update(self, seen, x, y):
        self.ax.set_xlim(0, len(self.x))
        self.line.set_data(self.x, self.y)
        self.ax.figure.canvas.draw()
        return self.line,


class Scope2(object):
    def __init__(self, ax, verbose=1, interval=1):
        #plot elements
        self.ax = ax
        self.x = [0]
        self.y = [0]
        self.markers = []
        self.line = Line2D(self.x, self.y, marker='D', markeredgecolor='r', markevery=self.markers)
        self.ax.add_line(self.line)
        self.ax.set_ylim(-.1, 1.1)
        self.ax.set_xlim(0, len(self.x))
        plt.ion()
        # redraw control parameters
        self.last_update = 0
        self.interval = interval
        self.verbose = verbose

    def _update_markers(self, markers):
        self.markers = markers

    def update(self, x, y):
        print(self.last_update)
        self.x = x
        self.y = y
        now = time.time()
        if self.verbose == 1:
            if (now - self.last_update) < self.interval:
                return
            else:
                # print('NONONO')
                self.ax.set_xlim(0, len(self.x))
                self.line.set_data(self.x, self.y)
                self.line.set_markevery(every = self.markers)
                # following line is time consuming
                self.ax.figure.canvas.draw()
                self.last_update = now
                return self.line,


class DummyCallback(Callback):
    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']
        self.nb_epoch = self.params['nb_epoch']
        self.nb_sample = self.params['nb_sample']
        self.batch_size = self.params['batch_size']
        self.i = 0
        self.x = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)
        self.y = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)

    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        self.seen = 0

    def on_batch_begin(self, batch, logs={}):
        if self.seen < self.params['nb_sample']:
            self.log_values = []

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        print('\n {}'.format(self.seen))
        self.x[self.i] = self.i
        self.y[self.i] = logs['loss']
        self.i += 1
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

    def on_epoch_end(self, epoch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

    def on_train_end(self, logs={}):
        print('training done')

class PlotLive(Callback):
    """Callback that outputs a live plot.
    """
    def __init__(self, scope):
        self.scope = scope
        self.markers = []

    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']
        self.nb_epoch = self.params['nb_epoch']
        self.nb_sample = self.params['nb_sample']
        self.batch_size = self.params['batch_size']
        self.i = 0
        self.x = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)
        self.y = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)
        self.total_seen = 0

    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        self.seen = 0

    def on_batch_begin(self, batch, logs={}):
        if self.seen < self.params['nb_sample']:
            self.log_values = []

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        self.total_seen += batch_size
        # print('\n {}'.format(self.seen))
        self.x[self.i] = self.i
        self.y[self.i] = logs['loss']
        self.i += 1
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # skip progbar update for the last batch;
        # will be handled by on_epoch_end
        if self.verbose and self.seen < self.params['nb_sample']:
            self.scope.update(self.x[:self.i], self.y[:self.i])

    def on_epoch_end(self, epoch, logs={}):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            # self.scope.update(self.seen)
            self.markers.append(self.i)
            self.scope._update_markers(self.markers)
            self.scope.update(self.x[:self.i], self.y[:self.i])


    def on_train_end(self, logs={}):
        print('training done')


def plotme():
    np.random.seed(0)
    x = np.linspace(0, 2 * math.pi, 100000)
    sine = np.sin(x)
    err = np.random.normal(0, 0.2, len(sine))
    y = sine + err

    n_conn = 60
    model = Sequential()
    model.add(Dense(output_dim=n_conn, input_dim=1))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=1))
    model.compile(loss='mean_squared_error', optimizer='sgd')

    X_train = np.array(x, ndmin=2).T
    Y_train = np.array(y, ndmin=2).T

    t0 = time.time()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    scope = Scope2(ax)
    ani = FuncAnimation(fig, scope.update)
    # ion needed in scope
    plt.show()
    history = PlotLive(scope=scope)
    dummy = DummyCallback()
    res = model.fit(X_train, Y_train, nb_epoch=20, verbose=1, batch_size=100, callbacks=[history])
    # res = model.fit(X_train, Y_train, nb_epoch=5, verbose=1, batch_size=100, callbacks=[dummy])


    t1 = time.time()
    elapsed = t1 - t0
    print('done {}'.format(elapsed))
    # done 4.033189535140991


if __name__ == '__main__':
    plotme()