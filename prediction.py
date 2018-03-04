import glob
import os
import numpy as np
import imageio
import pickle

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
        
    print ("Function is started")
    
    #convert img_folder into preds numpy vector  
    n_pictures = 0
    for filename in os.listdir(img_folder):
        n_pictures += 1
        print (n_pictures)

    images = np.zeros((n_pictures, 48, 48, 1))
        
    i = 0
    for filename in os.listdir(img_folder):
        picture_path = os.path.join("datasets/FER2013/Test", filename)
        images[i] = imageio.imread(picture_path)[:,:,0].reshape((48,48,1))
        #print (images[i])
        i += 1

    # feed pictures into NN and get predictions
    scores = model.loss(images)
    preds = np.argmax(scores, axis=1)

    
    return preds
    
test_fer_model("datasets/FER2013/Test")

'''
def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc
	

    ### End of code
    return preds
'''
