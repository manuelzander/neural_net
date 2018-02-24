import numpy as np

from src.classifiers import softmax
from src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)



def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    W = None
    b = None
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    W = weight_scale*np.random.randn(n_in, n_out).astype(dtype)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return W, b



class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10,
                 dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: A list giving the size of the input
        - num_classes: Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.

        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the Xavier
        initialisation (see manual).
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        W,b = random_init(input_dim, self.num_layers, weight_scale, dtype)

        # For loop to assign W and b values using Xavier initialisation for the defined number of layers of a network (i)
        for i in range(self.num_layers - 1):
            self.params['W' + str(i+1)] = W/np.sqrt(hidden_dims[i])
            self.params['b' + str(i+1)] = np.zeros(hidden_dims[i], dtype)

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.
        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"train": True, "p": dropout}
            if seed is not None:
                self.dropout_params["seed"] = seed
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        n_hidden_layer = self.num_layers - 1
        scores = X

        for i in range(n_hidden_layer):


            #LINEAR LAYER
            linear_cache['X%d' % (i + 1)] = scores
            linear_cache['W%d' % (i + 1)] = self.params['W%d' % (i + 1)]
            linear_cache['b%d' % (i + 1)] = self.params['b%d' % (i + 1)]
            scores = linear_forward(scores,
                                         self.params['W%d' % (i + 1)],
                                         self.params['b%d' % (i + 1)])

            #RELU LAYER
            relu_cache['X%d' % (i + 1)] = scores
            scores = relu_forward(scores)

            #DROPOUT LAYER
            if self.use_dropout:
                #dropout_cache['X%d' % (i + 1)] = scores
                scores, mask = dropout_forward(scores, self.dropout_params)
                dropout_cache['m%d' % (i + 1)] = mask

        #increase i counter by one for last layer
        i+= 1

        #LAST LINEAR LAYER
        linear_cache['X%d' % (i + 1)] = scores
        linear_cache['W%d' % (i + 1)] = self.params['W%d' % (i + 1)]
        linear_cache['b%d' % (i + 1)] = self.params['b%d' % (i + 1)]
        scores = linear_forward(scores,
                                 self.params['W%d' % (i + 1)],
                                 self.params['b%d' % (i + 1)])

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, dict()

        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        data_loss, d_logits = softmax(scores, y)
        reg_loss =  0
        for i in range(self.num_layers):
            reg_loss += 0.5 * self.reg * np.sum(self.params['W%d' % (i + 1)] ** 2)
            # Note: i + 1 because we start with W1, then W2, etc.
        loss = data_loss + reg_loss

        #Backprop for last layer
        X = linear_cache['X%d' % (num_layers + 1)]
        W = linear_cache['W%d' % (num_layers + 1)]
        b = linear_cache['b%d' % (num_layers + 1)]
        dout, grads['W%d' % (num_layers + 1)], grads['b%d' % (num_layers + 1)] = linear_backward(d_logits, X, W, b)

        #Include term for regularization
        grads['W%d' % (num_layers + 1)] += self.reg * self.params['W%d' % (num_layers + 1)]

        for i in range(self.num_layers-1, 0, -1):

            #Dropout backprop
            if self.dropout:
                mask = dropout_cache['m%d' % i]
                p = self.dropout_params["p"]
                train = self.dropout_params["train"]
                dout = dropout_backward(dout, mask, p, train)

            #Relu backprop
            dout = relu_backward(dout, relu_cache['X%d' % i])

            #Linear backprop
            X = linear_cache['X%d' % i]
            W = linear_cache['W%d' % i]
            b = linear_cache['b%d' % i]

            dout, grads['W%d' % i], grads['b%d' % i] = linear_backward(d_logits, X, W, b)

            #Include term for regularization
            grads['W%d' % i] += self.reg * self.params['W%d' % i]

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        return loss, grads
