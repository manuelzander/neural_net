import numpy as np


def softmax(logits, y):
	"""
    	Computes the loss and gradient for softmax classification.

    	Args:
    	- logits: A numpy array of shape (N, C)
    	- y: A numpy array of shape (N,). y represents the labels corresponding to
    	logits, where y[i] is the label of logits[i], and the value of y have a 
    	range of 0 <= y[i] < C

    	Returns (as a tuple):
    	- loss: Loss scalar
    	- dlogits: Loss gradient with respect to logits
    	"""
    	# initalise the loss and gradient to zero
	loss, dlogits = None, None
	"""
    	TODO: Compute the softmax loss and its gradient using no explicit loops
    	Store the loss in loss and the gradient in dW. If you are not careful
    	here, it is easy to run into numeric instability. Don't forget the
    	regularization!
    	"""
    	###########################################################################
    	#                           BEGIN OF YOUR CODE                            #
   	###########################################################################
	logits += -np.max (logits, axis=0)

	C = np.exp(logits)
	sum_of_C = np.sum(C, axis=0)

	gradient = C/sum_of_C
	
	loss = np.log(sum_of_C)



    	###########################################################################
    	#                            END OF YOUR CODE                             #
    	###########################################################################
	return loss, dlogits
