import collections
import math
import pyqtgraph as pg
import random
import time
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore


class DynamicPlotter():
    def __init__(self, sampleinterval=0.01, timewindow=10., size=(600,350), title=''):
        # Data stuff
        self._interval = int(sampleinterval*1000)
        self._bufsize = int(timewindow/sampleinterval)
        self.databuffer = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.givedata_buffer = None
        self.x = np.linspace(-timewindow, 0.0, self._bufsize)
        self.y = np.zeros(self._bufsize, dtype=np.float)

        # PyQtGraph stuff
        self.app = QtGui.QApplication([])
        self.title = 'Dynamic Plotting with PyQtGraph' if title == '' else title
        self.win = pg.GraphicsWindow()
        self.plt = self.win.addPlot(title=self.title)
        self.plt.resize(*size)
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'amplitude', 'V')
        self.plt.setLabel('bottom', 'time', 's')
        self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))

        # QTimer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateplot)
        self.timer.start(self._interval)

    def getdata(self):
        frequency = 0.5
        noise = random.normalvariate(0., 1.)
        new = 10.*math.sin(time.time()*frequency*2*math.pi) + noise
        return new

    def givedata(self, data):
        self.givedata_buffer = data

    def updateplot(self):
        # If we haven't received data to givedata_buffer
        if self.givedata_buffer == None:

            # Get your data from self.getdata()
            self.databuffer.append( self.getdata() )
        else:
            self.databuffer.append( self.givedata_buffer )

        self.y[:] = self.databuffer
        self.curve.setData(self.x, self.y)
        self.app.processEvents()

    def run(self):
        self.app.exec_()

    def data_wrapper(self, func, *args, **kwargs):
        def wrapped(*args, **kwargs):
            while True:
                res = func(*args, **kwargs)
                time.sleep(float(self._interval) / 1000)
                self.givedata(res)

        return wrapped


if __name__ == '__main__':
    # Import threading for having the thread to generate data
    import threading

    # Use this to see it in action
    plt = DynamicPlotter()

    # Example of how to give data to DynamicPlotter
    @plt.data_wrapper
    def data_gen(frequency=0.5, max_noise=1.):
        noise = random.normalvariate(0., max_noise)
        new = 10.*math.sin(time.time()*frequency*2*math.pi) + noise

        return new

    # Define the thread
    th = threading.Thread(target=data_gen,
                          kwargs={'frequency': 0.5,
                                  'max_noise': 1.})
    th.daemon = True

    # Finally when the set-up is ready, start everything
    th.start() # Start thread
    plt.run()  # Start plotting
