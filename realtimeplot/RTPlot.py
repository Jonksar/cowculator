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

    def updateplot(self):
        self.databuffer.append( self.getdata() )
        self.y[:] = self.databuffer
        self.curve.setData(self.x, self.y)
        self.app.processEvents()

    def run(self):
        self.app.exec_()

# Use this to see it in action
# x = DynamicPlotter()
# x.run()