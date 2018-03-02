import numpy as np
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data
import matplotlib.pyplot as plt
import pickle

#######################################################################
### LOAD DATA
#######################################################################

print("LOAD DATA")
#data = get_FER2013_data(25000, 3200, 4098)
data = get_FER2013_data(25000, 1000, 1000)

#######################################################################
### SET UP MODEL AND SOLVER
#######################################################################

H1, H2, reg = 100, 100, 0
#model = FullyConnectedNet([H1,H2], input_dim=48*48*3, num_classes=7, dropout=0, reg=reg)
model = FullyConnectedNet([H1,H2], input_dim=48*48*3, reg=reg)

'''
Example usage might look something like this:

data = {
  'X_train': # training data
  'y_train': # training labels
  'X_val': # validation data
  'y_val': # validation labels
}
model = MyAwesomeModel(hidden_size=100, reg=10)
solver = Solver(model, data,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-3,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=100,
                print_every=100)
'''

# Solver
optim_config = {'learning_rate' : 0.05} #default 1e-2
args = {
    'update_rule':"sgd_momentum",
    'optim_config':optim_config,
    'lr_decay':1,
    'batch_size':100,
    'num_epochs': 20#,
    #'verbose': False
}

solver = Solver(model, data, **args)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']

#print(X_train.shape)
#print(y_train.shape)
#print(X_val.shape)
#print(y_val.shape)

#print(model.params['W1'])
print("START TRAIN")
solver.train()
print("END TRAIN")
#print(model.params['W1'])

#######################################################################
### PLOT GRAPH
#######################################################################
'''
plt.subplot(2, 1, 1)
plt.title("Training loss")
plt.plot(solver.loss_history, "o")
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()
'''
#######################################################################
### SAVE AND LOAD MODEL
#######################################################################

with open('model.pkl', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('model.pkl', 'rb') as handle:
    loaded_model = pickle.load(handle)

#print("W1 after loading")
#print(loaded_model.params['W1'])
