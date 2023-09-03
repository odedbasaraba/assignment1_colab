from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    tmp_loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]
    # loop over data
    for i in range(N):
        zi = np.dot(X[i], W)

        # Shift unnormalized logits to prevent numerical instability
        zi -= np.max(zi)

        correct_class = zi[y[i]]
        tmp_loss = 0.0
        # loop over classes
        for j in range(C):
            tmp_loss += np.exp(zi[j])
        loss += np.log(np.exp(correct_class)/tmp_loss)
    loss = -loss / N
    reg_term = 0.5 * reg * np.sum(W * W) 
    loss += reg_term
    
    
    # Calculate the gradient with respect to weights (dW)
    for i in range(N):
        zi = np.dot(X[i], W)
        zi -= np.max(zi)
        exp_sum = np.sum(np.exp(zi))
        softmax_probs = np.exp(zi) / exp_sum
        
        for j in range(C):
            dW[:, j] += (softmax_probs[j] - (y[i] == j)) * X[i]

    # Average the gradient
    dW /= N
    # Add the gradient of the regularization term
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_trains = X.shape[0]
    z = np.dot(X, W)

    # Shift unnormalized logits to prevent numerical instability
    z -= np.max(z, axis=1, keepdims=True)

    exp_sum = np.sum(np.exp(z), axis= 1, keepdims=True)
    softmax_probs = np.exp(z) / exp_sum
  
    # Calculate the negative log likelihood of the correct class
    correct_logprobs = -np.log(softmax_probs[np.arange(num_trains), y])
    loss = np.sum(correct_logprobs) / num_trains
    loss += 0.5 * reg * np.sum(W * W)

    dZ = softmax_probs
    dZ[np.arange(num_trains), y] -= 1

    dW = np.dot(X.T, dZ) / num_trains
    dW += reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
