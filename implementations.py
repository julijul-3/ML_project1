import numpy as np 
import helpers as hp

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: initial weight vector
        max_iters: int, number of steps to run
        gamma: step-size
    Returns:
        w: last weight
        loss: corresponding loss value 
    """

    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        grad, e = hp.compute_gradient(y, tx, w)
        loss = hp.compute_mse(e)
        w = w - gamma * grad

    # store w and loss
    return loss, w

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
   
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
    y: numpy array of shape=(N, )
    tx: numpy array of shape=(N,2)
    initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
    max_iters: a scalar denoting the total number of iterations of SGD
    gamma: a scalar denoting the stepsize

    Returns:
    loss: last  loss value (scalar) for each iteration of SGD
    ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    
    w = initial_w

    for n_iter in range(max_iters):

        for y_batch, tx_batch in hp.batch_iter(y, tx, batch_size=1, num_batches=1):

            # compute a stochastic gradient and loss
            grad, _ = hp.compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = hp.compute_mse(y, tx, w)


    return loss, w


def least_squares(y, tx):
    """Least squares regression using normal equations
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    Returns:
        w: last weight
        loss: corresponding loss value 
    """

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations
    Args: 
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: regulization parameter
    Returns:
        w: last weight
        loss: corresponding loss value without penalty term
    """

def logistic_regression(y, tx, initial_w, max_iters, gamma):
     """Logistic regression using gradient descent or SGD (y ∈ {0, 1})
    Args: 
        y: numpy array of shape (N,), N is the number of samples. All values between 0 -1
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: initial weight vector
        max_iters: int, number of steps to run
        gamma: step-size
    Returns:
        w: last weight
        loss: corresponding loss value 
    """
     
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD (y ∈ {0, 1}, with regularization term λ∥w∥^2)
    Args: 
        y: numpy array of shape (N,), N is the number of samples. All values between 0 -1
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w: initial weight vector
        max_iters: int, number of steps to run
        gamma: step-size
    Returns:
        w: last weight
        loss: corresponding loss value without penalty term
    """