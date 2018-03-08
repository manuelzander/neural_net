import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Overfit the network with 50 samples of CIFAR-10
"""
############################################################################
#                            BEGIN OF YOUR CODE                            #
############################################################################

# Get data
data = get_CIFAR10_data(num_training=50)

# Create FC Net
H1, H2, reg = 100, 100, 0
model = FullyConnectedNet([H1,H2], reg=reg)

# Solver
optim_config = {'learning_rate' : 0.007} #default 1e-2
args = {'num_epochs':20, 'optim_config':optim_config}
solver = Solver(model, data, **args)

solver.train()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
