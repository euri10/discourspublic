from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np


class LivePlotHistory(Callback):
    def __init__(self, fp, update_on_batch=False, save=False):
        self.fp = fp
        self.update_on_batch = update_on_batch
        self.save = save
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        plt.show(block=False)
        self.x = []
        self.y = []

    def update(self, batch=None, logs=None):
        plt.gca().cla()
        if batch is not None:
            X = np.arange(len(self.losses))
            Y = np.array(self.losses)
            self.ax.plot(X, Y, '-rD', markevery=self.markers)
        else:
            pass
        plt.draw()
        plt.pause(0.01)
        if self.save:
            plt.savefig(self.fp + ".png")

    def on_train_begin(self, logs={}):
        self.losses = []
        self.markers = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        if self.update_on_batch:
            self.update(batch=batch, logs=logs)

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.markers.append(len(self.losses))
