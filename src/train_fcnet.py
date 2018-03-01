import numpy as np
import matplotlib.pyplot as plt

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

# Plot training and validation results
plt.subplot(2,1,1)
plt.title("Training loss")
plt.plot(solver.loss_history, "o")
plt.xlabel('Iteration')
plt.subplot(2,1,2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################
