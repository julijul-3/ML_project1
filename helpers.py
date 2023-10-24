import numpy as np
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt


### Compute 

def compute_mse(e):
    """Calculates mse for vector e"""

    return 1/2*np.mean(e**2)

def compute_gradient(y, tx, w):

    """Computes the gradient at w.

    Args:
    y: numpy array of shape=(N, )
    tx: numpy array of shape=(N,2)
    w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
    An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return compute_mse(e)

def compute_stoch_gradient(y, tx, w):
    
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.

    Args:
    y: numpy array of shape=(B, )
    tx: numpy array of shape=(B,2)
    w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
    A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.

    """
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array

    """
    return 1.0 / (1 + np.exp(-t))
   

  ### Loss for Logistic  : Sigmoïd 

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
 
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        a vector of shape (D, 1)

    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])
    return grad

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        a hessian matrix of shape=(D, D)
    """

    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1 - pred))
    return tx.T.dot(r).dot(tx) * (1 / y.shape[0])

def logistic_regression(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
        hessian: shape=(D, D)
    """
   
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian

### fonction du cours
def build_poly(x, degree):

    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    #raise NotImplementedError

    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss : mse loss, float 

    """

    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y,tx,w)
    
    return w, loss

### Load CSV


### ne fonctionne pas
def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids

def load_csv(data_path_x, data_path_y):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path_y, delimiter=",", skip_header=1, dtype=int)
    x_labels = np.genfromtxt(data_path_x, delimiter=",")
    x = x_labels[1:]
    labels = str(x_labels[0])
    ids = range(len(x))

    return x, y, ids, labels

def load_csv_1(data_path):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=int)
    x_labels = np.genfromtxt(data_path, delimiter=",", dtype = str)
    labels = (x_labels[0])
    ids = range(len(data))

    return  data , ids, labels



### Create CSV

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def create_csv(ids, labels, data, filename):
    print(len(ids))
    print(len(data))
    if len(ids) != len(data) or len(labels) != len(data[0]):
        raise ValueError("Length of IDs and values should match, and the number of labels should match the number of columns in the data.")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(labels)  # Writing header row with custom labels

        writer.writerows(data)
        #for j,row in zip(ids, data):
        #    writer.writerow(row)
    
    print(f"CSV file '{filename}' has been created successfully.")


### Plots

def plot_scatter(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(data)), data, alpha=0.5)
    plt.xlabel("Data Point Index")
    plt.ylabel("Values")
    plt.title("Scatter Plot of Values")
    plt.show()

### Cleaning

### Would be great to parametrize missing data if its -1 or nan or else ? 
def BMI_clean(bmi_data):

   # Create a mask to identify -1 values
    mask = (bmi_data != -1)

    # Calculate the mean of non-negative values
    mean_value = np.mean(bmi_data[mask])
    print(mean_value)

    # Replace -1 values with the mean
    bmi_data[~mask] = mean_value
    mean_value_2 = np.mean(bmi_data)
    print(mean_value_2)

    return bmi_data


def replace_nan_and_exception_with_mean(data, exception_values):
    
    # clean data for mean: filter all exceptions and negatives
    cleaned_data = data[~np.isin(data, exception_values) & (data > 0)]

    # Calculate the mean of non-nan values
    mean_value = np.nanmean(cleaned_data)

    # Replace bad values with the mean
    data[np.isnan(data)] = mean_value
    data[np.less(data, 0)] = mean_value
    data[np.isin(data, exception_values)] = mean_value
    
    return data

def replace_nan_and_exception_with_majority(data, exceptions):

    # Exclude negative , exception values and NaN values when counting occurrences
    valid_values = data[~np.isnan(data) & ~np.isin(data, exceptions) & (data >= 0)]
    value_counts = np.bincount(valid_values.astype(int))

    # Find the majority value
    majority_value = np.argmax(value_counts)

    # Replace exception & NaN with the majority value
    data[np.isnan(data)] = majority_value
    data[np.less(data, 0)] = majority_value
    data[np.isin(data, exceptions)] = majority_value

    return data

def data_clean(inputs):
    """
    inputs = (data, exceptions, with_majority)
    data:  data to clean
    exceptions:  values to remplace
    with_majority : (Bool) if True, remplace with majority, if false, remplace with mean
    """

    data, exceptions, with_majority = inputs
    
    if (with_majority):
        data = replace_nan_and_exception_with_majority(data,exceptions)
    else:
        data = replace_nan_and_exception_with_mean(data,exceptions)

    return data


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def clean_data(all_labels_list, labels_to_keep, dataset_to_clean):
    """
    Function to clean up the dataset while keeping only the wanted features
    Args:
        all_labels_list: (numpy array) every labels of the dataset, in order to find indices
        labels_to_keep: (str, numpy array 1x.., Bool) name of label, array of exception values, use_majority
        dataset_to_clean: (array NxD)
    """
    tab_train = []

    for input in labels_to_keep:
        label, exceptions, use_maj = input
        # find index of label
        id = np.where(all_labels_list == label)[0][0]
        # find data of this index
        data = dataset_to_clean[:, id]
    
        #clean up
        data_cleaned = data_clean((data, exceptions, use_maj))
        tab_train.append(data_cleaned)

    return np.array(tab_train).T