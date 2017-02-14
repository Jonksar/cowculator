import threading    # Import threading for having the thread to generate data
import random       # Random to generate noise to our data
import time         # Time to generate data
import math         # Time to generate data

# Import from path above
import os, sys
print os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

print sys.path
# Main star (importing plotting from directory above)
from plotting import DynamicPlotter

# Define dynamic plotter instance
plt = DynamicPlotter()

# Create function to give it data, and wrap it with instance.data_wrapper
@plt.data_wrapper
def data_gen(frequency=0.5, max_noise=1.):
    noise = random.normalvariate(0., max_noise)
    new = 10. * math.sin(time.time() * frequency * 2 * math.pi) + noise

    return new


# Define the data generation thread
th = threading.Thread(target=data_gen,
                      kwargs={'frequency': 0.5,
                              'max_noise': 1.})
th.daemon = True    # Kill when main thread dies

# Finally when the set-up is ready, start data_gen && plotting.
th.start()  # Start thread
plt.run()  # Start plotting
