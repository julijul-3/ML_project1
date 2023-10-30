import numpy as np
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt


def compute_mse(e):
    """Calculates mean squared error for vector e
    Args: 
        e: array of floats, error
    Returns:
        Mean square error for that vector (float)
    """

    return 1/2*np.mean(e**2)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        w: array of floats, weight value

    Returns:
        grad: array of floats (), gradient of the error at w.
        e : array of floats (), error at w
    """
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)

    return grad, e

def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        w: array of floats, weight value

    Returns:
        the value of the loss (a float), corresponding to w.
    """
    e = y - tx.dot(w)

    return compute_mse(e)

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w

    Args:
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        w: array of floats, weight value

    Returns:
        grad: array of floats (), gradient of the error at w.
        e : array of floats (), error at w
    """
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)

    return grad, e

#  Function taken from Ex02
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    Args:
        y: numpy array with the label values
        tx: numpy array with the features
        batch_size: int 
        num_batches: int, default 1
        shuffle: Bool, default True
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
        t: scalar or array
    Returns:
        scalar or array

    """

    return 1.0 / (1 + np.exp(-t))
   

  ### Loss for Logistic  : Sigmoïd 

def calculate_loss_logistic(y, tx, w):
    """ Loss by negative log likelihood.

    Args:
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        w: array of floats, weight value
    Returns:
        loss: float, a non-negative loss corresponding to w
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
 
    pred = sigmoid(tx.dot(w))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)) 
    
    return loss
    
def calculate_gradient_logistic(y, tx, w):
    """ Gradient for logistic regression.

    Args:
        y: array (N,), N is the number of samples, prediction labels.
        tx: array (N,D), D is the number of features, data points.
        w: array of floats, weight value
    Returns:
        grad: a vector of shape (D, 1), the gradient

    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])
    return grad

### Helpers for ridge regression
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    Creates a polynomial version of the data x, of degree degree

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

### Load and write CSV

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

### Cleaning our data

def replace_nan_and_exception_with_mean(data, exception_values):
    """ 
    Computes the mean of the feature and replace all nan, exception_values, and negative values with it
    Normalizes the data
    Args:
    data: numpy array, data to be cleaned (one feature)
    exception_values: numpy array, values to be replaced
    Returns:
    cleaned data, numpy array
    """
    
    #filter all exceptions and negatives and find mean
    cleaned_data = data[~np.isin(data, exception_values) & (data > 0)]
    mean_value = np.nanmean(cleaned_data)

    # Replace bad values with the mean
    data[np.isnan(data)] = mean_value
    data[np.less(data, 0)] = mean_value
    data[np.isin(data, exception_values)] = mean_value
    
    # normalise
    stds = np.std(data,axis=0)
    data = (data-mean_value)/stds

    return data

def replace_nan_and_exception_with_majority(data, exceptions):
    """
    Finds the value that is the most frequent and replaces the nan, negative, and exception values with it
    Args:
    data: numpy array, one feature to be cleaned
    exceptions: numpy array, values to be replaced
    Returns: cleaned data, numpy array
    """

    # Exclude negative, exception values and nan values when counting occurrences
    valid_values = data[~np.isnan(data) & ~np.isin(data, exceptions) & (data >= 0)]
    value_counts = np.bincount(valid_values.astype(int))

    majority_value = np.argmax(value_counts)

    # Replace exception and nan and negatives with the majority value
    data[np.isnan(data)] = majority_value
    data[np.less(data, 0)] = majority_value
    data[np.isin(data, exceptions)] = majority_value

    return data

def replace_value(value, data):
    """
    Replaces the value 'value' by zero in the data
    Args:
        value: int, value to replace
        data: numpy array size (1, N) 
    Returns: 
        the dataset with corrections
    """
    cleaned_data = np.where(data == value, 0, data)
    return cleaned_data

def replace_exceptions(inputs):

    """
    Unpacks inputs and call the correct clean method depending on the type
    Args:
    inputs = (data, exceptions, with_majority)
    data:  data to clean
    exceptions:  values to remplace
    with_majority : (Bool) if True, remplace with majority, if false, remplace with mean
    Returns: cleaned data, numpy array
    """

    data, exceptions, with_majority = inputs
    
    if (with_majority):
        data = replace_nan_and_exception_with_majority(data,exceptions)
    else:
        data = replace_nan_and_exception_with_mean(data,exceptions)

    return data


def clean_data(all_labels_list, labels_to_keep, dataset_to_clean):
    """
    Function to clean up the dataset while keeping only the wanted features
    Args:
        all_labels_list: (numpy array) every labels of the dataset, in order to find indices
        labels_to_keep: (str, numpy array 1x.., Bool) name of label, array of exception values, use_majority
        dataset_to_clean: (array NxD)
    Returns:
        the cleaned dataset
    """
    tab_train = []

    for input in labels_to_keep:
        label, exceptions, use_maj, replaced = input
        # find index of label
        id = np.where(all_labels_list == label)[0][0]
        # find data of this index
        data = dataset_to_clean[:, id]

        if (replaced!=None):
            data = replace_value(replaced, data)
    
        #clean up
        data_cleaned = replace_exceptions((data, exceptions, use_maj))
        tab_train.append(data_cleaned)

    return np.array(tab_train).T

### prediction of labels

def predict_labels_mse(weights, data):
    """Generates class predictions given weights, and a test data matrix
    Predicts -1/1 and separates at threshold 0
    Args: 
        weights: numpy array (N,D) weights of trained model
        data: numpy array (D, ) cleaned test data
    Returns: 
        y_pred: numpy array of predictions -1/1
    """
    y_pred = np.dot(data, weights)
    print("y pred : '{}'.".format(y_pred))
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def predict_labels_logistic(weights, data):
    """Generates class predictions for logistic weights
    Computes probabilities and separates at threshold 0.5
    Args:
        weights: numpy array (N, D) weights from trained model
        data: numpy array (D, ) cleaned test dataset
    Returns: 
        y_pred: numpy array, predictions of labels, -1/1
    """
    z = np.dot(data, weights)
    probabilities = sigmoid(z)
    y_pred = np.copy(probabilities)
    print("y pred : '{}'.".format(y_pred))
    y_pred[np.where(probabilities <= 0.5)] = -1
    y_pred[np.where(probabilities > 0.5)] = 1
    return y_pred

## helpers for cross validation

def split_train_test(y, x, proportion):
    """
    Splits the train test for cross validation

    Args:
    y (array-like): the labels
    x (array-like): the data points
    proportion (float): percentage of data used for train (between 0-1)


    Returns: 
    Array-like: x_train, the data points used for training
    Array_like: y_train, the labels used for training
    Array-like: x_test, the data points used for testing
    Array-like: y_test, the labels used for testing
    """
    index = int(np.ceil(len(x)*proportion))
    if index>= len(x) :
        print("index too large")
    x_train = x[:index]
    x_test = x[index:]
    y_train = y[:index]
    y_test = y[index:]
    return x_train, y_train, x_test, y_test

def measure_accuracy(y, y_pred):
    """
    Measures the accuracy of a given prediction

    Args:
    y (array-like): the labels 
    y_pred (array-like): the predictions

    Returns: 
    Float: the accuracy
    """
    return sum(a==b for a,b in zip(y, y_pred))/len(y)

def measure_f1_score(y, y_pred):
    """
    Measures the f1 score of a given prediction

    Args:
    y (array-like): the labels 
    y_pred (array-like): the predictions

    Returns: 
    Float: f1 score
    """
    true_pos = sum(a==b and b==1 for a,b in zip(y,y_pred))
   # print("True pos : '{}'.".format(true_pos))
    false_neg = sum(a!=b and b==-1 for a,b in zip(y,y_pred))
   # print("False neg : '{}'.".format(false_neg))
    false_pos = sum(a!=b and b==1 for a,b in zip(y,y_pred))
    #print("False pos : '{}'.".format(false_pos))
    return (true_pos/(true_pos+1/2 * (false_neg+false_pos)))