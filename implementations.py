import numpy as np 
import helpers as hp

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    
    """Linear regression using gradient descent
    Args: 
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        initial_w: initial weight vector
        max_iters: int, number of iterations to run
        gamma: a scalar denoting the stepsize
    Returns:
        w: array of floats, last weight value
        loss: float, last loss value 
    """

    # Initialization of parameters
    w = initial_w
    loss = hp.compute_loss(y, tx, w)
    grad, e = hp.compute_gradient(y, tx, w)

    for n_iter in range(max_iters):

        w = w - gamma * grad
        grad, e = hp.compute_gradient(y, tx, w)
        loss = hp.compute_mse(e)

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
   
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        initial_w: initial weight vector
        max_iters: int, number of iterations to run
        gamma: a scalar denoting the stepsize
    Returns:
        w: array of floats, last weight value
        loss: float, last loss value 
    """

    # Initialization of parameters
    w = initial_w
    loss = hp.compute_loss(y, tx, w)

    for n_iter in range(max_iters):

        for y_batch, tx_batch in hp.batch_iter(y, tx, batch_size=1, num_batches=1):

            grad, _ = hp.compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = hp.compute_loss(y, tx, w)


    return w, loss

def least_squares(y, tx):

    """Least squares regression.
    Args:
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
    Returns:
        w: array of floats, last weight value
        loss: float, last loss value
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = hp.compute_loss(y, tx, w)

    return w, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression.

    Args:
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        lambda_: scalar, regularization parameter.
    Returns:
        w: array of floats, last weight value
        loss: float, last loss value
    """
   
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = hp.compute_loss(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression.

    Args: 
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        initial_w: initial weight vector
        max_iters: int, number of iterations to run
        gamma: a scalar denoting the stepsize
    Returns:
        w: array of floats, last weight value
        loss: float, last loss value 
    """

    # Initialization of parameters
    w = initial_w
    loss = hp.calculate_loss_logistic(y, tx, w)
    grad = hp.calculate_gradient_logistic(y, tx, w)

    for iter in range(max_iters):

        w -= gamma * grad
        loss = hp.calculate_loss_logistic(y, tx, w)
        grad = hp.calculate_gradient_logistic(y, tx, w)

    return w, np.array(loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression.

    Args: 
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        lambda_: scalar, regularization parameter.
        initial_w: initial weight vector
        max_iters: int, number of iterations to run
        gamma: a scalar denoting the stepsize
    Returns:
        w: array of floats, last weight value
        loss: float, last loss value
    """

    # Initialization of parameters
    w = initial_w
    loss = hp.calculate_loss_logistic(y, tx, w) 
    gradient = hp.calculate_gradient_logistic(y, tx, w) + 2*lambda_*w

    for i in range(max_iters):

        w-=gamma*gradient
        loss = hp.calculate_loss_logistic(y, tx, w)
        gradient = hp.calculate_gradient_logistic(y, tx, w) + 2*lambda_*w

    return w, loss