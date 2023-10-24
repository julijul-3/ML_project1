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
    loss = hp.compute_loss(y, tx, w)
    grad, e = hp.compute_gradient(y, tx, w)
    for n_iter in range(max_iters):
        w = w - gamma * grad
        grad, e = hp.compute_gradient(y, tx, w)
        loss = hp.compute_mse(e)
        

    # store w and loss
    return w, loss

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
    loss = hp.compute_loss(y, tx, w)

    for n_iter in range(max_iters):

        for y_batch, tx_batch in hp.batch_iter(y, tx, batch_size=1, num_batches=1):

            # compute a stochastic gradient and loss
            grad, _ = hp.compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = hp.compute_loss(y, tx, w)


    return w, loss

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = hp.compute_loss(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
   
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = hp.compute_loss(y, tx, w)
    return w, loss

### LOGITIC REG ###

def sigmoid(t):
    """
    The sigmoid function modified to be numerically stable

    Parameters:
    t (array-like): the input of the sigmoid function

    Returns:
    Array-like: the array formed by applying the sigmoid function
    """
    sigs = np.zeros(len(t))
    sigs[np.where(t >= 0)] = 1 / (1 + np.exp(-t[t >= 0]))
    sigs[np.where(t < 0)] = np.exp(t[t < 0]) / (1 + np.exp(t[t < 0]))
    return sigs

# Found here: https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python


def negative_log_likelihood(y, tx, w):
    """
    Calculates the negative log likelihood

    Parameters:     
    y (array-like): the labels
    tx (array-like): the data points
    w (array-like): the weights
    
    Returns:
    Array-like: the negative log likelihood
    """
    sigs = sigmoid(tx@w)
    loss = y.T@np.log(sigs, where=sigs > 1e-50) + (1 - y).T@np.log(1 - sigs, where=1-sigs > 1e-50)
    return -loss

def gradient_negative_log_likelihood(y, tx, w):
    """
    Calculates the gradient of negative log likelihood
    
    Parameters:     
    y (array-like): the labels
    tx (array-like): the data points
    w (array-like): the weights
    
    Returns:
    Array-like: the gradient of the negative log likelihood
    """
    return tx.T@(sigmoid(tx@w)-y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using the negative log likelihood as a cost function

    Parameters:
    y (array-like): the labels
    tx (array-like): the data points
    initial_w (array-like): the initial weight vector
    max_iters (int): the number of steps to run
    gamma (float): the step-size

    Returns: 
    Array-like: the weights
    Float: the loss/negative log likelihood
    """
    w = initial_w
    loss = None
    for i in range(max_iters):
        loss = negative_log_likelihood(y, tx, w)
        gradient = gradient_negative_log_likelihood(y, tx, w)
        w-=gamma*gradient
    return w, loss



### possiblement pas mettre dans la loss la penalite mais la mettre dans le gradient descent 
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
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic linear
    for iteration in range(max_iters):
        # get loss and update w.
        loss, gradient, w = hp.gradient_descent_step(y, tx, w, gamma, lambda_, mode='logistic')
        # log info
        if iteration % 100 == 0:
            print("Current iteration={i}, loss={loss}".format(
                i=iteration, loss=loss))
            print("||d|| = {d}".format(d=np.linalg.norm(gradient)))
        # converge criterion
        losses.append(loss)
        # print("Current iteration={i}, loss={l}, ||d|| = {d}".format(i=iter, l=loss, d=np.linalg.norm(gradient)))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    # visualization
    print("loss={l}".format(l=hp.compute_loss(y, tx, w)))

    return w, losses[-1]