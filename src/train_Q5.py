import numpy as np
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_FER2013_data
import matplotlib.pyplot as plt
import pickle

#######################################################################
### LOAD DATA
#######################################################################

num_training = 20000
num_validation = 3000
#num_training = 25709
#num_validation = 3000
num_test = 0

print("LOAD DATA")
data = get_FER2013_data(num_training, num_validation, num_test)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']

#Check data format for FER2013 (1, 48, 48)
assert X_train.shape == (num_training, 1, 48, 48)
assert y_train.shape == (num_training,)
assert X_val.shape == (num_validation, 1, 48, 48)
assert y_val.shape == (num_validation,)

#######################################################################
### SET UP MODEL AND SOLVER
#######################################################################

#New dictionary to store classification rates
classification_rate_cache = dict()
max_classification_rate = 0.0;
best_method = None;
best_model = None;
best_solver = None;
solvers = [];

#learning_rates = [1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5]
learning_rates = [1e-3] #Use this one, turned out to be th best

#Loop through learning rates specified above
for lr in learning_rates:
    print('*************************************************************')
    print('LEARNING RATE: %f' % lr)

    #Loop through no of neurons
    for number_neurons in range(200, 800, 600):
        print('****************************')
        print('NO OF NEURONS: %d' % number_neurons)

        no_neurons_layer1 = number_neurons
        no_neurons_layer2 = number_neurons
        #no_neurons_layer3 = number_neurons

        #Set up of 4 models, 1 hidden layer / 2 hidden layers with and without dropout
        #model_one_layer = FullyConnectedNet([no_neurons_layer1], input_dim=1*48*48, reg=reg, dtype=np.float64)
        model_two_layers = FullyConnectedNet([no_neurons_layer1,no_neurons_layer2], input_dim=1*48*48, reg=0.0, dtype=np.float64)
        model_two_layers_L2 = FullyConnectedNet([no_neurons_layer1,no_neurons_layer2], input_dim=1*48*48, reg=0.005, dtype=np.float64)
        #model_three_layers = FullyConnectedNet([no_neurons_layer1,no_neurons_layer2,no_neurons_layer3], input_dim=1*48*48, reg=reg, dtype=np.float64)
        #model_one_layer_withdropout = FullyConnectedNet([no_neurons_layer1], input_dim=1*48*48, dropout=0.5, reg=reg, dtype=np.float64)
        #model_two_layers_withdropout = FullyConnectedNet([no_neurons_layer1,no_neurons_layer2], input_dim=1*48*48, dropout=0.05, reg=reg, dtype=np.float64)

        #Set values for solver
        optim_config = {'learning_rate' : lr} #Note that default is 1e-2
        args = {
            'update_rule':"sgd_momentum", #Note that this is 0.9 default in optim.py
            'optim_config':optim_config,
            'lr_decay':0.95,
            'batch_size':100,
            'num_epochs':40,
            'verbose': False
        }

        #Create solver instances for each model
        #solver_one_layer = Solver(model_one_layer, data, **args)
        solver_two_layers = Solver(model_two_layers, data, **args)
        solver_two_layers_L2 = Solver(model_two_layers_L2, data, **args)
        #solver_three_layers = Solver(model_three_layers, data, **args)
        #solver_one_layer_withdropout = Solver(model_one_layer_withdropout, data, **args)
        #solver_two_layers_withdropout  = Solver(model_two_layers_withdropout, data, **args)
        #solvers.append(solver_two_layers)

        #Train models with solver instances and store classification rates
        '''
        solver_one_layer.train()
        classification_rate_cache['ONE_L%dNEUR%fLR' % (number_neurons, lr)] = solver_one_layer.best_val_acc
        if(solver_one_layer.best_val_acc > max_classification_rate):
            best_model = solver_one_layer.model
            best_solver = solver_one_layer
        '''

        solver_two_layers.train()
        classification_rate_cache['TWO_L%dNEUR%fLR' % (number_neurons, lr)] = solver_two_layers.best_val_acc
        if(solver_two_layers.best_val_acc > max_classification_rate):
            best_model = solver_two_layers.model
            best_solver = solver_two_layers

        solver_two_layers_L2.train()
        classification_rate_cache['TWO_L%dNEUR%fLR_L2' % (number_neurons, lr)] = solver_two_layers_L2.best_val_acc
        if(solver_two_layers_L2.best_val_acc > max_classification_rate):
            best_model = solver_two_layers_L2.model
            best_solver = solver_two_layers_L2

        '''
        solver_three_layers.train()
        classification_rate_cache['THREE_L%dNEUR%fLR' % (number_neurons, lr)] = solver_three_layers.best_val_acc
        if(solver_three_layers.best_val_acc > max_classification_rate):
            best_model = solver_three_layers.model
            best_solver = solver_three_layers

        solver_one_layer_withdropout.train()
        classification_rate_cache['ONE_L%dNEUR%fLR_DROP' % (number_neurons, lr)] = solver_one_layer_withdropout.best_val_acc
        if(solver_one_layer_withdropout.best_val_acc > max_classification_rate):
            best_model = solver_one_layer_withdropout.model
            best_solver = solver_one_layer_withdropout

        solver_two_layers_withdropout.train()
        classification_rate_cache['TWO_L%dNEUR%fLR_DROP' % (number_neurons, lr)] = solver_two_layers_withdropout.best_val_acc
        if(solver_two_layers_withdropout.best_val_acc > max_classification_rate):
            best_model = solver_two_layers_withdropout.model
            best_solver = solver_two_layers_withdropout
        '''

#print(classification_rate_cache)
print("{:<60} {:<60}".format('Method','Classification Rate'))
for i in classification_rate_cache:
    print("{:<60} {:<60}".format(i,classification_rate_cache[i]))

    if (classification_rate_cache[i] > max_classification_rate):
        max_classification_rate = classification_rate_cache[i]
        best_method = i

print('Max classification rate: %f' % max_classification_rate)
print('Best method: %s' % best_method)

#######################################################################
### PLOT GRAPH FOR USING DROPOUT
#######################################################################

plt.subplot(2, 2, 1)
plt.title("Training loss")
plt.plot(solver_two_layers.loss_history, "o", markersize=0.5)
plt.xlabel('Iteration')

plt.subplot(2, 2, 2)
plt.title('Accuracy')
plt.plot(solver_two_layers.train_acc_history, '-o', label='train', markersize=0.5)
plt.plot(solver_two_layers.val_acc_history, '-o', label='val', markersize=0.5)
plt.xlabel('Epoch')

plt.subplot(2, 2, 3)
plt.title("Training loss")
plt.plot(solver_two_layers_L2.loss_history, "o", markersize=0.5)
plt.xlabel('Iteration')

plt.subplot(2, 2, 4)
plt.title('Accuracy')
plt.plot(solver_two_layers_L2.train_acc_history, '-o', label='train', markersize=0.5)
plt.plot(solver_two_layers_L2.val_acc_history, '-o', label='val', markersize=0.5)
plt.xlabel('Epoch')

plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()

#######################################################################
### PLOT GRAPH FOR LEARNING RATE OPIMIZATION
#######################################################################
'''
plt.subplot(2, 1, 1)
plt.title("Training loss")
plt.gca().set_ylim([0,2.5])
for i in range(0, len(solvers)):
    plt.plot(solvers[i].loss_history, "o", label='%f lr' % learning_rates[i], markersize=0.5)
plt.xlabel('Iteration')
#plt.legend(loc='upper right')

plt.subplot(2, 2, 3)
plt.title('Accuracy (training)')
for i in range(0, len(solvers)):
    plt.plot(solvers[i].train_acc_history, '-o', label='%f lr' % learning_rates[i], markersize=3)
plt.xlabel('Epoch')
#plt.legend(loc='lower right')

plt.subplot(2, 2, 4)
plt.title('Accuracy (validation)')
for i in range(0, len(solvers)):
    plt.plot(solvers[i].val_acc_history, '-o', label='%f lr' % learning_rates[i], markersize=3)
plt.xlabel('Epoch')
plt.legend(loc='lower right')
'''
'''
plt.subplot(2, 2, 3)
plt.title('F1 measure')
for i in range(0, len(solvers)):
    plt.bar(solvers[i].final_F1, '-o', label='val')
plt.xlabel('Learning rates')
plt.legend(loc='upper right')

plt.subplot(2, 2, 4)
plt.title('Precision rate')
for i in range(0, len(solvers)):
    plt.bar(solvers[i].final_precision, '-o', label='val')
plt.xlabel('Learning rates')
plt.legend(loc='upper right')
'''
#plt.gcf().set_size_inches(15, 12)
#plt.show()

#######################################################################
### PLOT GRAPH
#######################################################################
'''
plt.subplot(2, 1, 1)
plt.title("Training loss")
plt.plot(best_solver.loss_history, "o")
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(best_solver.train_acc_history, '-o', label='train')
plt.plot(best_solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(best_solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()
'''
#######################################################################
### SAVE AND LOAD MODEL
#######################################################################

with open('best_model.pkl', 'wb') as handle:
    pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('best_solver.pkl', 'wb') as handle2:
    pickle.dump(best_solver, handle2, protocol=pickle.HIGHEST_PROTOCOL)

'''
with open('model.pkl', 'rb') as handle:
    loaded_model = pickle.load(handle)

#print("W1 after loading")
#print(loaded_model.params['W1'])
'''
