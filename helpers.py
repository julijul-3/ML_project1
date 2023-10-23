import numpy as np
import csv
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

### Load CSV

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    # ids = x[:, 0].astype(np.int)
    ids = range(len(x))
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return y, input_data, ids

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

def load_data(sub_sample=True, add_outlier=False):
    """Load data and convert it to the metric system."""
    path_dataset = "height_weight_genders.csv"
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
    height = data[:, 0]
    weight = data[:, 1]
    gender = np.genfromtxt(
        path_dataset,
        delimiter=",",
        skip_header=1,
        usecols=[0],
        converters={0: lambda x: 0 if b"Male" in x else 1},
    )
    # Convert to metric system
    height *= 0.025
    weight *= 0.454
    return height, weight, gender

### Create CSV

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def create_csv_1(ids, labels, data, filename):
    """
   
    """
    if len(ids) != len(data):
        raise ValueError("Length of IDs and values should be the same")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', labels])  # Writing header row with custom label

        for id, value in zip(ids,data):
            writer.writerow([id, value])
    
    print(f"CSV file '{filename}' has been created successfully.")   

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


def replace_nan_and_exception_with_mean(data, exception_value):
    # Create a mask to identify exception values
    mask = (data != exception_value)
    
    # Calculate the mean of non-(-1) values
    mean_value = np.mean(data[mask])
    
    # Calculate the mean of non-NaN values
    mean_value_nan = np.nanmean(data)

    # Check for NaN values and print a message
    if np.isnan(data).any():
        print("There are NaN values")

    # Replace NaN values with the mean
    data[np.isnan(data)] = mean_value_nan
    
    # Replace -1 values with the mean
    data[~mask] = mean_value
    
    return data

def replace_nan_and_exception_with_majority(data, exception):

    # Exclude -1 , exception value and NaN values when counting occurrences
    valid_values = data[~np.isnan(data) & (data != exception)& (data != -1)]
    value_counts = np.bincount(valid_values.astype(int))

    # Find the majority value
    majority_value = np.argmax(value_counts)

    # Replace exception & NaN with the majority value
    data[np.isnan(data)] = majority_value
    data[data == exception] = majority_value
    data[data == -1] = majority_value
    return data

def data_clean(data, exception):
    
    if (exception == 9):
        data = replace_nan_and_exception_with_majority(data,exception)

    if((exception == -1) or (exception == 99900)):
        data = replace_nan_and_exception_with_mean(data,exception)

    return data