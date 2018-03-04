import glob
import os
import numpy as np
import imageio
import pickle
from src.utils.solver import construct_confusion_matrix, prediction_measures

# TO DO: change the path of the model
def test_fer_model(img_folder, model_path="model.pkl"):
    
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
        picture_path = os.path.join("datasets/FER2013/Test", filename)
        images[i] = imageio.imread(picture_path)[:,:,0].reshape((48,48,1))
        i += 1

    # feed pictures into NN and get predictions
    scores = model.loss(images)
    preds = np.argmax(scores, axis=1)
    
    return preds

# TO DO: change the path of the model
# Tests the model with the secret dataset
def test_deep_fer_model(img_folder, model_path="model.pkl"):
    
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
        picture_path = os.path.join("datasets/FER2013/Test", filename)
        images[i] = imageio.imread(picture_path)[:,:,0].reshape((48,48,1))
        i += 1

    # feed pictures into NN and get predictions
    scores = model.loss(images)
    preds = np.argmax(scores, axis=1)
    
    return preds


# Reports the confusion matrix and prediction measures using the FER2013 dataset
def test_Q6_FER2013(prediction_pickle_path, model_path="model.pkl"):
    
    with open(prediction_pickle_path, 'rb') as handle:
        data = pickle.load(handle) 

    with open(model_path, 'rb') as handle:
        model = pickle.load(handle) 

        
    scores = model.loss(data['X_test'])
    preds = np.argmax(scores, axis=1)

    confusion_matrix = construct_confusion_matrix(preds, data['y_test'])

    classification_rate, recall_vector, precision_vector, f1_measure = prediction_measures(confusion_matrix)
    return confusion_matrix, classification_rate, recall_vector, precision_vector, f1_measure

confusion_matrix, classification_rate, recall_vector, precision_vector, f1_measure = test_Q6_FER2013("FER2013_data.pickle", model_path="model.pkl")

print ("Confusion matrix: ")
print (confusion_matrix, "\n")
print ("Classification rate: ")
print (classification_rate, "\n")
print ("F1 measure: ")
print (f1_measure)

