import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    HC = filter_size  # conv filter height
    WC = filter_size
    F = num_filters
    P = (filter_size-1) / 2 # padding
    SC = 1
    conv_h = (H+2*P-HC) / SC + 1
    conv_w = (W+2*P-WC) / SC + 1
    HP = 2
    WP = 2  # pool filter size 2*2
    SP = 2
    pool_h = 1 + (conv_h - HP) / SP
    pool_w = 1 + (conv_w - WP) / SP

    # mind not to use np.random.rand!
    self.params['W1'] = weight_scale * np.random.randn(F, C, HC, WC)
    self.params['b1'] = np.zeros(F)
    # what is the size of fc layer?
    # the input of affine layer is (N, F, pool_h, pool_w) and should be reshaped into (N,F*pool_h*pool_w)
    # weights should be (F*pool_h*pool_w, H)
    self.params['W2'] = weight_scale * np.random.randn(F*pool_h*pool_w, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)

    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    SC = conv_param['stride']
    PC = conv_param['pad']
    HP = pool_param['pool_height']
    WP = pool_param['pool_width']
    SP = pool_param['stride']

    # conv forward with conv_relu_forward_fast
    out, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    # fully-connected forward
    # print X.shape, out.shape
    # print W2.shape, b2.shape
    # print out.reshape([out.shape[0], np.prod(out.shape[1:])]).shape
    # (50, 3, 32, 32)(50, 32, 16, 16)
    # (8192, 100)(100, )
    # (50, 8192)
    # h1, cache_h1 = affine_relu_forward(out.reshape([out.shape[0], np.prod(out.shape[1:])]), W2, b2)
    h1, cache_h1 = affine_relu_forward(out, W2, b2)
    scores, cache_scores = affine_forward(h1, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2) + 0.5 * self.reg * np.sum(W3 * W3)
    loss = data_loss + reg_loss

    # backprop
    dh2, dW3, db3 = affine_backward(dscores, cache_scores)
    dh1, dW2, db2 = affine_relu_backward(dh2, cache_h1)
    dX, dW1, db1 = conv_relu_pool_backward(dh1, cache_conv)

    # note to add L2 regularization!!!
    grads['W1'] = dW1 + self.reg * W1
    grads['W2'] = dW2 + self.reg * W2
    grads['W3'] = dW3 + self.reg * W3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


class TwoConvLayerCNN(object):
    """
    A deeper cnn compared to ThreeLayerConvnet.
    conv - relu - conv - relu - 2x2 max pool - affine - affine - relu -affine - softmax
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.mem = {}
        self.reg = reg
        self.dtype = dtype

        # variables
        C, H, W = input_dim
        F = num_filters
        HC = filter_size  # conv filter height
        WC = filter_size
        SC = 1
        P = (filter_size-1) / 2 # the value of padding make sure that the spatial size stays unchanged after convolution
        conv_h = (H + 2 * P - HC) / SC + 1
        conv_w = (W + 2 * P - WC) / SC + 1
        HP = 2
        WP = 2  # pool filter size 2*2
        SP = 2
        pool_h = 1 + (conv_h - HP) / SP
        pool_w = 1 + (conv_w - WP) / SP

        self.params['W0'] = weight_scale * np.random.randn(F, C, HC, WC)
        self.params['b0'] = np.zeros(F)
        self.params['W1'] = weight_scale * np.random.randn(F, F, HC, WC)
        self.params['b1'] = np.zeros(F)
        # what is the size of fc layer?
        # the input of affine layer is (N, F, pool_h, pool_w) and should be reshaped into (N,F*pool_h*pool_w)
        # weights should be (F*pool_h*pool_w, H)
        self.params['W2'] = weight_scale * np.random.randn(F * pool_h * pool_w, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W0, b0 = self.params['W0'], self.params['b0']
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # two conv-relu layer
        scores, cache_conv0 = conv_relu_forward(X, W0, b0, conv_param)
        scores, cache_conv1 = conv_relu_pool_forward(scores, W1, b1, conv_param, pool_param)

        h1, cache_h1 = affine_relu_forward(scores, W2, b2)
        scores, cache_scores = affine_forward(h1, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W0 * W0) + 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2) + 0.5 * self.reg * np.sum(
            W3 * W3)
        loss = data_loss + reg_loss

        # backprop
        dh2, dW3, db3 = affine_backward(dscores, cache_scores)
        dh1, dW2, db2 = affine_relu_backward(dh2, cache_h1)
        dc1, dW1, db1 = conv_relu_pool_backward(dh1, cache_conv1)
        dX, dW0, db0 = conv_relu_backward(dc1, cache_conv0)

        # note to add L2 regularization!!!
        grads['W0'] = dW0 + self.reg * W0
        grads['W1'] = dW1 + self.reg * W1
        grads['W2'] = dW2 + self.reg * W2
        grads['W3'] = dW3 + self.reg * W3
        grads['b0'] = db0
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3

        return loss, grads


class ThreeConvLayerCNN(object):
    """
    A deeper cnn compared to ThreeLayerConvnet.
    conv - relu - conv - relu - conv - relu - 2x2 max pool - affine - affine - relu -affine - softmax
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.mem = {}
        self.reg = reg
        self.dtype = dtype

        # variables
        C, H, W = input_dim
        F = num_filters
        HC = filter_size  # conv filter height
        WC = filter_size
        SC = 1
        P = (filter_size-1) / 2 # the value of padding make sure that the spatial size stays unchanged after convolution
        conv_h = (H + 2 * P - HC) / SC + 1
        conv_w = (W + 2 * P - WC) / SC + 1
        HP = 2
        WP = 2  # pool filter size 2*2
        SP = 2
        pool_h = 1 + (conv_h - HP) / SP
        pool_w = 1 + (conv_w - WP) / SP
        # conv parameters
        self.params['W0'] = weight_scale * np.random.randn(F, C, HC, WC)
        self.params['b0'] = np.zeros(F)
        self.params['W1'] = weight_scale * np.random.randn(F, F, HC, WC)
        self.params['b1'] = np.zeros(F)
        self.params['W2'] = weight_scale * np.random.randn(F, F, HC, WC)
        self.params['b2'] = np.zeros(F)
        # fc parameters
        self.params['W3'] = weight_scale * np.random.randn(F * pool_h * pool_w, hidden_dim)
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b4'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        W0, b0 = self.params['W0'], self.params['b0']
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # two conv-relu layer
        scores, cache_conv0 = conv_relu_forward(X, W0, b0, conv_param)
        scores, cache_conv1 = conv_relu_forward(scores, W1, b1, conv_param)
        scores, cache_conv2 = conv_relu_pool_forward(scores, W2, b2, conv_param, pool_param)

        h1, cache_h1 = affine_relu_forward(scores, W3, b3)
        scores, cache_scores = affine_forward(h1, W4, b4)

        if y is None:
            return scores

        loss, grads = 0, {}

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W0 * W0) + 0.5 * self.reg * np.sum(W1 * W1) + 0.5 * self.reg * np.sum(W2 * W2) + 0.5 * self.reg * np.sum(W3 * W3) + 0.5 * self.reg * np.sum(
            W4 * W4)
        loss = data_loss + reg_loss

        # backprop
        dh2, dW4, db4 = affine_backward(dscores, cache_scores)
        dh1, dW3, db3 = affine_relu_backward(dh2, cache_h1)
        dc2, dW2, db2 = conv_relu_pool_backward(dh1, cache_conv2)
        dc1, dW1, db1 = conv_relu_backward(dc2, cache_conv1)
        dX, dW0, db0 = conv_relu_backward(dc1, cache_conv0)

        # note to add L2 regularization!!!
        grads['W0'] = dW0 + self.reg * W0
        grads['W1'] = dW1 + self.reg * W1
        grads['W2'] = dW2 + self.reg * W2
        grads['W3'] = dW3 + self.reg * W3
        grads['W4'] = dW4 + self.reg * W4
        grads['b0'] = db0
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        grads['b4'] = db4

        return loss, grads
pass
