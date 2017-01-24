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

class Scope2(object):
    def __init__(self, ax, verbose=1, interval=1, save=False, filename='myplot.png' ):
        # ax2 will hold the accuracy and ax the loss
        self.ax = ax
        self.ax2 = self.ax.twinx()
        # b_x_loss plot the batch losses on ax (left axis)
        # e_x_loss plot the epoch losses on ax (left axis)
        self.b_x_loss = [0]
        self.b_y_loss = [0]
        self.e_x_loss = [0]
        self.e_y_loss = [0]
        # b_x_acc plot the batch losses on ax2 (right axis)
        # e_x_acc plot the epoch losses on ax2 (right axis)
        self.b_x_acc = [0]
        self.b_y_acc = [0]
        self.e_x_acc = [0]
        self.e_y_acc = [0]
        # e_x_val_acc plot the epoch validation accuracy on ax2 (right axis)
        # e_x_val_loss plot the epoch validation loss on ax (right axis)
        self.e_x_val_acc = [0]
        self.e_y_val_acc = [0]
        self.e_x_val_loss = [0]
        self.e_y_val_loss = [0]
        # unused atm, could be for each epoch
        self.markers = []
        # define the loss lines
        self.line_b_loss = Line2D(self.b_x_loss, self.b_y_loss, linewidth=.1, linestyle=':', color='b')
        self.line_e_loss = Line2D(self.e_x_loss, self.e_y_loss, linewidth=.3, color='b')
        self.line_e_val_loss = Line2D(self.e_x_val_loss, self.e_y_val_loss, linewidth=.3, color='g')
        # define the accuracy lines
        self.line_b_acc = Line2D(self.b_x_acc, self.b_y_acc, linewidth=.1, linestyle=':', color='r')
        self.line_e_acc = Line2D(self.e_x_acc, self.e_y_acc, linewidth=.3, color='r')
        self.line_e_val_acc = Line2D(self.e_x_val_acc, self.e_y_val_acc, linewidth=.3, color='g')
        # add the lines to relevant axes, losses goes to ax and accuracy to ax2
        self.ax.add_line(self.line_b_loss)
        self.ax.add_line(self.line_e_loss)
        self.ax.add_line(self.line_e_val_loss)
        self.ax2.add_line(self.line_b_acc)
        self.ax2.add_line(self.line_e_acc)
        self.ax2.add_line(self.line_e_val_acc)
        # set the plot limits, raw
        self.ax.set_ylim(-.1, 1.1)
        self.ax.set_xlim(0, len(self.b_x_loss))
        # make it interactive, important
        plt.ion()
        # redraw control parameters
        self.last_update = 0
        self.interval = interval
        self.verbose = verbose
        #save elements
        self.save = save
        self.filename = filename

    def _update(self, b_x_loss, b_y_loss, e_x_loss, e_y_loss, b_x_acc, b_y_acc, e_x_acc, e_y_acc, e_x_val_acc, e_y_val_acc, e_x_val_loss, e_y_val_loss, force=False):
        print(self.last_update)
        self.b_x_loss = b_x_loss
        self.b_y_loss = b_y_loss
        self.e_x_loss = e_x_loss
        self.e_y_loss = e_y_loss
        self.b_x_acc = b_x_acc
        self.b_y_acc = b_y_acc
        self.e_x_acc = e_x_acc
        self.e_y_acc = e_y_acc
        self.e_x_val_acc = e_x_val_acc
        self.e_y_val_acc = e_y_val_acc
        self.e_x_val_loss = e_x_val_loss
        self.e_y_val_loss = e_y_val_loss
        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return
            else:
                print('force: {}'.format(force))
                self.ax.set_xlim(0, len(self.b_x_loss))
                self.line_b_loss.set_data(self.b_x_loss, self.b_y_loss)
                self.line_e_loss.set_data(self.e_x_loss, self.e_y_loss)
                self.line_b_acc.set_data(self.b_x_acc, self.b_y_acc)
                self.line_e_acc.set_data(self.e_x_acc, self.e_y_acc)
                self.line_e_val_acc.set_data(self.e_x_val_acc, self.e_y_val_acc)
                self.line_e_val_loss.set_data(self.e_x_val_loss, self.e_y_val_loss)
                # following line is time consuming, hence the check for time update, increase interval to redraw less frequently
                self.ax.figure.canvas.draw()
                self.last_update = now
                return self.line_b_loss, self.line_e_loss, self.line_b_acc, self.line_e_acc, self.line_e_val_acc, self.line_e_val_loss

    def _save(self):
        if self.save:
            plt.savefig(self.filename)


class DummyCallback(Callback):
    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']
        self.nb_epoch = self.params['nb_epoch']
        self.nb_sample = self.params['nb_sample']
        self.batch_size = self.params['batch_size']
        self.i = 0
        self.b_x_loss = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)
        self.b_y_loss = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)

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
        self.b_x_loss[self.i] = self.i
        self.b_y_loss[self.i] = logs['loss']
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
    """Callback that outputs a live matplotlib plot.
    """
    def __init__(self, scope, plot_validate=False):
        self.scope = scope
        self.markers = []
        self.plot_validate = plot_validate

    def on_train_begin(self, logs={}):
        self.verbose = self.params['verbose']
        self.nb_epoch = self.params['nb_epoch']
        self.nb_sample = self.params['nb_sample']
        self.batch_size = self.params['batch_size']
        print('verbose{} nb_epoch{} nb_sample{} batch_size{}'.format(self.verbose, self.nb_epoch, self.nb_sample, self.batch_size))
        self.i = 0
        self.e = 0

        self.b_x_loss = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)
        self.b_y_loss = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)
        self.e_x_loss = np.zeros(self.nb_epoch)
        self.e_y_loss = np.zeros(self.nb_epoch)

        #if self.params['do_validation'] and self.plot_validate:
        self.b_x_acc = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)
        self.b_y_acc = np.zeros((int((self.nb_sample / self.batch_size)) + 1) * self.nb_epoch)

        self.e_x_acc = np.zeros(self.nb_epoch)
        self.e_y_acc = np.zeros(self.nb_epoch)

        self.e_x_val_acc = np.zeros(self.nb_epoch)
        self.e_y_val_acc = np.zeros(self.nb_epoch)

        self.e_x_val_loss = np.zeros(self.nb_epoch)
        self.e_y_val_loss = np.zeros(self.nb_epoch)

        self.e_x_pred = np.zeros(self.nb_epoch)
        self.e_y_pred = np.zeros(self.nb_epoch)



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
        # print('\n {}'.format(self.seen))
        if self.params['do_validation'] and self.plot_validate:
            self.b_x_acc[self.i] = self.i
            self.b_y_acc[self.i] = logs['acc']
        self.b_x_loss[self.i] = self.i
        self.b_y_loss[self.i] = logs['loss']
        self.i += 1
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        if self.verbose and self.seen < self.params['nb_sample']:
            # self.scope._update(self.b_x_loss[:self.i], self.b_y_loss[:self.i])
            # self.scope._update(self.b_x_loss[:self.i], self.b_y_loss[:self.i], self.e_x_loss[:self.i], self.e_y_loss[:self.i])
            self.scope._update(self.b_x_loss[:self.i],
                               self.b_y_loss[:self.i],
                               self.e_x_loss[:self.e],
                               self.e_y_loss[:self.e],
                               self.b_x_acc[:self.i],
                               self.b_y_acc[:self.i],
                               self.e_x_acc[:self.e],
                               self.e_y_acc[:self.e],
                               self.e_x_val_acc[:self.e],
                               self.e_y_val_acc[:self.e],
                               self.e_x_val_loss[:self.e],
                               self.e_y_val_loss[:self.e],
                               )

    def on_epoch_end(self, epoch, logs={}):
        if self.params['do_validation'] and self.plot_validate:
            self.e_x_acc[self.e] = self.i
            self.e_y_acc[self.e] = logs['acc']

            self.e_x_val_acc[self.e] = self.i
            self.e_y_val_acc[self.e] = logs['val_acc']

            self.e_x_val_loss[self.e] = self.i
            self.e_y_val_loss[self.e] = logs['val_loss']

        self.e_x_loss[self.e] = self.i
        self.e_y_loss[self.e] = logs['loss']

        self.e += 1
        if self.verbose:
            if epoch < self.nb_epoch-1:
                self.scope._update(self.b_x_loss[:self.i],
                                   self.b_y_loss[:self.i],
                                   self.e_x_loss[:self.e],
                                   self.e_y_loss[:self.e],
                                   self.b_x_acc[:self.i],
                                   self.b_y_acc[:self.i],
                                   self.e_x_acc[:self.e],
                                   self.e_y_acc[:self.e],
                                   self.e_x_val_acc[:self.e],
                                   self.e_y_val_acc[:self.e],
                                   self.e_x_val_loss[:self.e],
                                   self.e_y_val_loss[:self.e],
                                   )
            else:
                self.scope._update(self.b_x_loss[:self.i],
                                   self.b_y_loss[:self.i],
                                   self.e_x_loss[:self.e],
                                   self.e_y_loss[:self.e],
                                   self.b_x_acc[:self.i],
                                   self.b_y_acc[:self.i],
                                   self.e_x_acc[:self.e],
                                   self.e_y_acc[:self.e],
                                   self.e_x_val_acc[:self.e],
                                   self.e_y_val_acc[:self.e],
                                   self.e_x_val_loss[:self.e],
                                   self.e_y_val_loss[:self.e],
                                   force=True
                                   )

    def on_train_end(self, logs={}):
        self.scope._save()
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
    scope = Scope2(ax, save=True)
    ani = FuncAnimation(fig, scope._update)
    # ion needed in scope
    plt.show()
    history = PlotLive(scope=scope)
    # dummy = DummyCallback()
    res = model.fit(X_train, Y_train, nb_epoch=4, verbose=1, batch_size=100, callbacks=[history])
    # res = model.fit(X_train, Y_train, nb_epoch=4, verbose=1, batch_size=100, callbacks=[dummy])
    t1 = time.time()
    elapsed = t1 - t0
    print('done {}'.format(elapsed))



if __name__ == '__main__':
    plotme()