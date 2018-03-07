#from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform
import imageio
import pickle

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def get_FER2013_data(num_training, num_validation, num_test, subtract_mean=True):

    X_train, y_train, X_test, y_test = load_FER2013()

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test)) # Our test set is not seperate from training?
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def load_FER2013():

    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    data_source = 'datasets/FER2013_data.pickle'
    data_path = os.path.join(root_path, data_source)
    
    print (data_path)
    print ("Test")
    
    #with open('/vol/bitbucket/osk17/FER2013_data.pickle', 'rb') as handle:
    with open(data_path, 'rb') as handle:
        s = pickle.load(handle)

    '''
    f2 = open('FER2013_data.pickle', 'rb')
    s = pickle.load(f2)
    f2.close()
    '''

    X_train = s['X_train']
    y_train = s['y_train']
    X_test = s['X_test']
    y_test = s['y_test']

    return X_train, y_train, X_test, y_test

def collect_FER2013_data(filepath):

    file = open(filepath)
    #first call to ignore headings
    first_line = file.readline()
    train_count  = 0
    test_count = 0

    for line in file:
        if 'Train' in line:
            train_count += 1

        elif 'Test' in line:
            test_count += 1
    file.close()

    file = open(filepath)

    first_line = file.readline()

    X_train = np.empty((train_count,48,48,1))
    X_test = np.empty((test_count,48,48,1))
    y_train = []
    y_test = []

    train_count = 0
    test_count = 0

    i = 0
    for line in file:
        print (line)

        #put training data into X_train
        if 'Train' in line:
            #Getting y y_train and appending to list
            y = line[-2]
            y_train.append(y)
            print (y)

            #Getting X_train and putting into X_train numpy array
            pic_path = line.split(',')[0]
            path = os.path.join('/vol/bitbucket/395ML_NN_Data/datasets/FER2013', pic_path)
            X_train[train_count] = imageio.imread(path)[:,:,0].reshape((48,48,1))
            train_count += 1

            #i += 1
            #if(i == 10):
            #    break;


        elif 'Test' in line:

            y = line[-2]

            if y == ',':
                y = line[-1]

            y_test.append(y)

            print (y)
            #Getting X_train and putting into X_train numpy array
            pic_path = line.split(',')[0]

            path = os.path.join('/vol/bitbucket/395ML_NN_Data/datasets/FER2013', pic_path)
            X_test[test_count] = imageio.imread(path)[:,:,0].reshape((48,48,1))
            test_count += 1

            #print ("WHY")
            #print (X_test.shape)
            #print (len(y_test))

    y_train = np.array(y_train).astype("int64")
    y_test = np.array(y_test).astype("int64")


    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test
    }

    f = open('/homes/mmz216/ML_Assignment_2/src/utils/FER2013_data.pickle', 'wb') #/vol/bitbucket/osk17
    pickle.dump(data, f)
    f.close()

    return
#############################################################################

#collect_FER2013_data('/vol/bitbucket/395ML_NN_Data/datasets/FER2013/labels_public.txt')
#a,b,c,d = load_FER2013()

#x = get_FER2013_data(num_training=4, num_validation=2, num_test=0,
#                     subtract_mean=True)

#print (a.dtype)
#print (b.dtype)
#print (c.dtype)
#print (d.dtype)

#print (x['X_train'][0][0])
#print (x['X_test'].shape)
#print (x['X_val'].shape)
#print (x['y_val'].shape)
#print (x['y_train'].shape)
#print (x['y_test'].shape)
