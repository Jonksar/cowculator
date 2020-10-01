import math
import numpy as np
import os
import random
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from cowculator import plotting


def plot_1d():
    # Use this to see it in action
    plt = plotting.DynamicPlotter(sampleinterval=0.01, timewindow=10.)

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

def plot_np():
    # Use this to see it in action
    plt = plotting.DynamicPlotterNumpy(sampleinterval=0.01, timewindow=10.)

    # Example of how to give data to DynamicPlotter
    @plt.data_wrapper
    def data_gen(n_dim=10):
        noise = np.array([random.normalvariate(0., .2) for i in range(n_dim)])
        data = np.array([float(i) * math.sin(time.time() * 2 *math.pi) for i in range(n_dim)]) + noise

        return data

    th = threading.Thread(target=data_gen)
    th.daemon=True

    th.start()
    plt.run()


def numpy_sin_plot():
    pass

if __name__ == '__main__':
    print ("This is the plotting example of KoalaTools toolkit.\nEnter a number in order to see that particular example.\n\t(1) Fast real time plotting of sin + noise\n\t(2) Slower real time plotting of multiple lines sin + noise\n")

    choice = input("Enter your choice:")

    if "1" in choice:
        plot_1d()
    elif "2" in choice:
        plot_np()
