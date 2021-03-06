import glob
import os
import numpy as np
import imageio
import pickle
from src.utils.solver import construct_confusion_matrix, prediction_measures
import keras
from keras.models import load_model
from src.utils.data_utils import get_FER2013_data

def test_fer_model(img_folder, model_path="saved_models/best_model_Q5.pkl"):
    
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

    subtract_mean = True
    
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_path = os.path.join(root_path, model_path)
    
    # Load model
    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)        
        
    # Sorting the folder names in lexographic order
    images_list = os.listdir(img_folder)
    images_list.sort()

    n_pictures = 0
    for filename in images_list:
        n_pictures += 1

    # Convert img_folder into preds numpy vector
    images = np.zeros((n_pictures, 48, 48, 1))
        
    i = 0
    for filename in images_list:
        #check this one
        picture_path = os.path.join(img_folder, filename)
        images[i] = imageio.imread(picture_path)[:,:,0].reshape((48,48,1))
        i += 1
        
    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(images, axis=0)
        images -= mean_image

    images = images.transpose(0, 3, 1, 2).copy()
        
    # Feed pictures into NN and get predictions
    scores = model.loss(images)
    preds = np.argmax(scores, axis=1)

    return preds

def test_deep_fer_model(img_folder, model_path="saved_models/best_model_Q6.h5"):

    
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
        
    # IMAGE PREPROCESSING
    # NO subtracting mean applied since model was trained without it
    subtract_mean = True
    if subtract_mean:
        mean_image = np.mean(images, axis=0)
        images -= mean_image

    # divide by 255
    images = images/255
    
    # Load and initialize the model
    model = load_model(model_path)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


    # feed pictures into NN and get predictions
    preds = model.predict(images)

    # Convert predictions from for example [0,0,0,1,0,0,0] to [4]
    preds = preds.argmax(1)

    return preds


# Q5 - Tests performance on test dataset to approximate test on secret dataset
# Return prediction vector
'''
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = "datasets/Test"
picture_path = os.path.join(root_path, data_path)

preds = test_fer_model(picture_path)

# Test the model

data_short_path = "datasets/FER2013_data.pickle"
data_test = os.path.join(root_path, data_short_path)

with open(data_test, 'rb') as handle:
    data = pickle.load(handle)

y_test = data['y_test']

confusion_matrix = construct_confusion_matrix(y_test, preds)
classification_rate, _, _, f1_measure = prediction_measures(confusion_matrix)

print ("Confusion matrix: ")
print (confusion_matrix, "\n")
print ("Classification rate: ")
print (classification_rate, "\n")
print ("F1 measure: ")
print (f1_measure)
'''

# Q6 - Test performance on test dataset to approximate test on secret dataset
# Returns prediction vector
'''
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_path = "datasets/Test"
data_path = os.path.join(root_path, data_path)

data_short_path = "datasets/FER2013_data.pickle"
data_test = os.path.join(root_path, data_short_path)

preds = test_deep_fer_model(data_path)

with open(data_test, 'rb') as handle:
    data = pickle.load(handle)

y_test = data['y_test']

confusion_matrix = construct_confusion_matrix(y_test, preds)
classification_rate, _, _, f1_measure = prediction_measures(confusion_matrix)

print ("Confusion matrix: ")
print (confusion_matrix, "\n")
print ("Classification rate: ")
print (classification_rate, "\n")
print ("F1 measure: ")
print (f1_measure)
'''
