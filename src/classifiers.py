import numpy as np
import math


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

	#Small term to include later against divide by zero errors
	epsi = np.finfo(np.float64).eps

	num_classes = logits.shape[1]
	num_train = logits.shape[0]

	logits += - np.max(logits)
	logits_exp = np.exp(logits)
	sum_of_exp = np.sum(logits_exp, axis=1, keepdims=True)

	p = logits_exp/sum_of_exp

	correct_class_probabilities = -np.log(p[range(num_train), y] + epsi)
	loss = np.sum(correct_class_probabilities)/num_train

	#calculate gradient
	gradient = logits_exp / sum_of_exp
	gradient[range(num_train), y] += -1.0
	dlogits = gradient/num_train

	###########################################################################
	#                            END OF YOUR CODE                             #
	###########################################################################
	return loss, dlogits
