import numpy as np 

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
    return None

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
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