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

	#Darcy's code

	'''
	logits += -np.max (logits, axis=0)

	C = np.exp(logits)
	sum_of_C = np.sum(C, axis=0)

	gradient = C/sum_of_C

	loss = np.log(sum_of_C)
	'''

	#print(logits.shape)
	#print(logits[:,0])
	#logits = logits.transpose()
	#logits = logits.reshape((1, -1))
	#print(logits.shape)
	#print(logits[:,0])

	num_classes = logits.shape[1]
	num_train = logits.shape[0]

	logits += - np.max(logits)
	logits_exp = np.exp(logits) #+ math.log(0.5)))
	sum_of_exp = np.sum(logits_exp, axis=1, keepdims=True)
	p = logits_exp/sum_of_exp

	print(num_train)
	print(y)


	correct_class_probabilities = -np.log(p[range(num_train), y])
	loss = np.sum(correct_class_probabilities)/num_train

        #calculate gradient
	gradient = logits_exp / sum_of_exp
	gradient[range(num_train), y] += -1.0
	dlogits = gradient/num_train

        #HOW DO WE GET W? Don't need W - they just want us to do normalisation for numerical stability (did this)
	#loss += 0.5 * 0.1 * np.sum(W*W)

	#probabilities[range(num_train),y] -= 1
	#dlogits = X.T.dot(probabilities) / num_train
	'''
	loss = loss - score[y,np.arange(num_train)]
	loss = np.sum(loss) / float(num_train) + 0.5 * reg * np.sum(W*W)

	Grad = exp_score / sum_exp_score_col
	Grad[y,np.arange(num_train)] += -1.0
	dW = Grad.dot(X.T) / float(num_train) + reg*W
	'''

	###########################################################################
	#                            END OF YOUR CODE                             #
	###########################################################################
	return loss, dlogits
