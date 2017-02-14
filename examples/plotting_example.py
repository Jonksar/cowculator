import random
import time
import math
import threading

# Doing extraordinary magic in order to import from a folder above
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
print sys.path
import plotting

def d_sin_plot():
    # Use this to see it in action
    plt = plotting.DynamicPlotter()

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


def numpy_sin_plot():
    pass

if __name__ == '__main__':
    d_sin_plot()
