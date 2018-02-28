import numpy as np

from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data

"""
TODO: Use a Solver instance to train a TwoLayerNet that achieves at least 50% 
accuracy on the validation set.
"""
#############################################################################
#                           BEGIN OF YOUR CODE                              #
#############################################################################

#Get data 
data = get_CIFAR10_data()

# Create FC Net
H1, H2, reg = 100, 100, 0
two_layer_net = FullyConnectedNet([H1,H2], reg=reg)

# Solver
optim_config = {'learning_rate' : 2e-3} #default 1e-2

args = {
    'update_rule':"sgd",
    'optim_config':optim_config,
    'lr_decay':0.90,
    'batch_size':100,
    'num_epochs': 20
}
solver = Solver(two_layer_net, data, **args)

solver.train()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
