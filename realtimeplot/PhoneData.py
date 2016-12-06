from RTPlot import DynamicPlotter
import numpy as np
import collections
import socket
import threading
import math

UDP_IP = "192.168.1.36"
UDP_PORT = 5555


def exp_smooth(x_new, x_old, alpha=0.85):
    return alpha * x_old + x_new * (1 - alpha)


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


class DataPlot(DynamicPlotter):
    def __init__(self):
        DynamicPlotter.__init__(self, sampleinterval=0.01, timewindow=2, title="Phone data plotting")

        self.p = Phone()
        th = threading.Thread(target=self.p)
        th.daemon = True
        th.start()

        # 3 acceleration buffers for phone
        self.databuffer_A = [collections.deque([0.0] * self._bufsize, self._bufsize) for i in range(3)]
        self.databuffer_G = [collections.deque([0.0] * self._bufsize, self._bufsize) for i in range(3)]

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
        phone_data = self.p.data

        A = np.array([phone_data['acc_x'] / 9.81, phone_data['acc_y'] / 9.81, phone_data['acc_z'] / 9.81])
        G = np.array([math.degrees(phone_data['gyr_x']), math.degrees(phone_data['gyr_y']), math.degrees(phone_data['gyr_z'])])

        # xyz
        for i in range(3):
            self.databuffer_A[i].append(A[i])
            self.databuffer_G[i].append(G[i])

        res = exp_smooth(A, self.last_data, alpha=0.75)
        self.last_data = res

        fin_array = res
        fin_array = np.array([fin_array[i] if i < len(fin_array) else 0 for i in range(self.n_curves)])

        return np.hstack((self.plotData, np.array(fin_array).reshape((-1, 1))))

    def updateplot(self):
        self.plotData = self.getdata()
        self.plotData = self.plotData[:, -self._bufsize:]

        for i in range(self.n_curves):
            self.plot_lines[i].setData(self.x, self.plotData[i, -self._bufsize:])

        self.frames_since_start += 1

        self.app.processEvents()

x = DataPlot()
x.run()
