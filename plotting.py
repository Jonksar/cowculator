import pyqtgraph as pg
import random
import time
import numpy as np
import collections
import socket
import threading
import math
from pyqtgraph.Qt import QtGui, QtCore

def exp_smooth(x_new, x_old, alpha=0.85):
    return alpha * x_old + x_new * (1 - alpha)


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


UDP_IP = "192.168.1.36"
UDP_PORT = 5555

# IMU WIRELESS PARSER FOR ANDROID PHONE
class Phone:
    def __init__(self):
        # Because socks are nice
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Internet, UDP
        self.sock.bind((UDP_IP, UDP_PORT))
        self.data = {'acc_x': 0, 'acc_y': 0, 'acc_z': 0, 'gyr_x': 0, 'gyr_y': 0, 'gyr_z': 0, 'time': 0}

    def __call__(self, *args, **kwargs):
        while True:
            data, addr = self.sock.recvfrom(1024)
            data = map(float, data.split(','))
            self.data['time'] = data.pop(0)

            while len(data) != 0:
                identifier = data.pop(0)

                if identifier == 3:
                    self.data['acc_x'] = data.pop(0)
                    self.data['acc_y'] = data.pop(0)
                    self.data['acc_z'] = data.pop(0)

                elif identifier == 4:
                    self.data['gyr_x'] = data.pop(0)
                    self.data['gyr_y'] = data.pop(0)
                    self.data['gyr_z'] = data.pop(0)

                elif identifier == 5:
                    data.pop(0)
                    data.pop(0)
                    data.pop(0)

                else:
                    raise ValueError("identifier of %d is not known" % identifier)


class DynamicPlotterNumpy(DynamicPlotter):
    def __init__(self, sampleinterval=0.01, timewindow=2, title="Phone data plotting"):
        DynamicPlotter.__init__(self, sampleinterval=sampleinterval, timewindow=timewindow, title=title)

        # 3 acceleration buffers for phone
        self.databuffer_A = np.zeros((3, self._bufsize))
        self.databuffer_G = np.zeros((3, self._bufsize))

        # Plotting buffers
        self.n_curves = 15
        self.plotData = np.zeros((self.n_curves, self._bufsize), dtype=np.float)

        self.curve_colors = [(255, 255, 255), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 255, 0),
                             (0, 0, 255), (0, 255, 255), (0, 255, 255), (255, 0, 155), (100, 255, 0), (100, 100, 100),
                             (0, 255, 255), (255, 0, 155), (100, 255, 0), (100, 100, 100)]

        self.plot_lines = [self.plt.plot(x=self.x, y=self.plotData[i, :], pen=self.curve_colors[i % len(self.curve_colors)]) for i in range(self.n_curves)]

        # Plot settings
        self.plt.setYRange(-4, 4, padding=0.05)

        # Convenience
        self.frames_since_start = 0

        # For smoothing data coming in
        self.last_data = np.zeros(3)

    def getdata(self):
        pass

    def updateplot(self):
        self.plotData = self.getdata()
        self.plotData = self.plotData[:, -self._bufsize:]

        for i in range(self.n_curves):
            self.plot_lines[i].setData(self.x, self.plotData[i, -self._bufsize:])

        self.frames_since_start += 1

        self.app.processEvents()

if __name__ == '__main__':

    if input() == "":
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
    else:

        # Initialize instances
        plt_np = DynamicPlotterNumpy()  # plotting device
        p = Phone()                     # phone device

        @plt_np.data_wrapper
        def data_gen():
            phone_data = p.data

            res_data = np.array([phone_data['acc_x'] / 9.81,
                                 phone_data['acc_y'] / 9.81,
                                 phone_data['acc_z'] / 9.81,
                                 math.degrees(phone_data['gyr_x']),
                                 math.degrees(phone_data['gyr_y']),
                                 math.degrees(phone_data['gyr_z'])])
            """
            fin_array = res_data
            fin_array = np.array([fin_array[i] if i < len(fin_array) else 0 for i in range(self.n_curves)])

            return np.hstack((self.plotData, np.array(fin_array).reshape((-1, 1))))
            """

        th = threading.Thread(target=p) # we are calling p.__call__() from thread
        th.daemon = True                # Kill when main thread dies


        th.start()
        plt_np.run()
