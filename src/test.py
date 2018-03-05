import glob
import os
import numpy as np
import imageio
import pickle
from src.utils.solver import construct_confusion_matrix, prediction_measures
import keras
from keras.models import load_model


# TO DO: change the path of the model
def test_fer_model(img_folder, model_path="saved_models/model.pkl"):
    
    """
    Given a folder with images, load the images and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None

    ### Start your code here

    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_path = os.path.join(root_path, model_path)
    
    #Load model
    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)        
        
    #sorting the folder names in lexographic order
    images_list = os.listdir(img_folder)
    images_list.sort()
    
    n_pictures = 0
    for filename in images_list:
        n_pictures += 1

    #convert img_folder into preds numpy vector
    images = np.zeros((n_pictures, 48, 48, 1))
        
    i = 0
    for filename in images_list:
        #check this one
        picture_path = os.path.join(images_list, filename)
        images[i] = imageio.imread(picture_path)[:,:,0].reshape((48,48,1))
        i += 1

    # feed pictures into NN and get predictions
    scores = model.loss(images)
    preds = np.argmax(scores, axis=1)
    
    return preds

# TO DO: change the path of the model
# Tests the model with the secret dataset
def test_deep_fer_model(img_folder, model_path="saved_models/keras_cifar10_trained_model.h5"):
    
    """
    Given a folder with images, load the images and your best model to predict
    the facial expression of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in
    img_folder.
    """
    preds = None

    ### Start your code here
    
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_path = os.path.join(root_path, model_path)

    #getting folder names and sorting the folder names in lexographic order
    images_list = os.listdir(img_folder)
    images_list.sort()
    
    n_pictures = 0
    for filename in images_list:
        n_pictures += 1

    #convert img_folder into preds numpy vector
    images = np.zeros((n_pictures, 48, 48, 1))
        
    i = 0
    for filename in images_list:
        picture_path = os.path.join(img_folder, filename)
        images[i] = imageio.imread(picture_path)[:,:,0].reshape((48,48,1))
        i += 1

    images = images.transpose(0, 3, 1, 2).copy()
        
    
    # Load and initialize the model
    model = load_model(model_path)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


    # feed pictures into NN and get predictions
    preds = model.predict(images)

    # Convert predictions from [0,0,0,1,0,0,0] to [4]
    preds = preds.argmax(1)

    return preds


# Reports the confusion matrix and prediction measures using the FER2013 dataset
def test_Q6_FER2013(prediction_pickle_path, model_path="saved_models/keras_cifar10_trained_model.h5"):

    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_path = os.path.join(root_path, model_path)

    
    prediction_pickle_path = os.path.join(root_path, prediction_pickle_path)
    
    # Load data from pickle file
    with open(prediction_pickle_path, 'rb') as handle:
        data = pickle.load(handle) 

    X_test = data['X_test']
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    

    # Load and initialize the model
    model = load_model(model_path)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    preds = model.predict(X_test)

    # Convert predictions from [0,0,0,1,0,0,0] to [4]
    preds = preds.argmax(1)
    
    confusion_matrix = construct_confusion_matrix(data['y_test'], preds)
    
    classification_rate, recall_vector, precision_vector, f1_measure = prediction_measures(confusion_matrix)
    return confusion_matrix, classification_rate, recall_vector, precision_vector, f1_measure




confusion_matrix, classification_rate, _, _, f1_measure = test_Q6_FER2013("FER2013_data.pickle")

'''
print ("Confusion matrix: ")
print (confusion_matrix, "\n")
print ("Classification rate: ")
print (classification_rate, "\n")
print ("F1 measure: ")
print (f1_measure)
'''

### QUESTION 6 TEST ###
pred  = test_deep_fer_model("datasets/FER2013/Train/", model_path="saved_models/keras_cifar10_trained_model.h5")

print (pred)

