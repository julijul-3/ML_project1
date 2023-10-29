import numpy
from helpers import *
from implementations import *

"""
Training and testing our model on the Behavioral Risk Factor Surveillance
 System dataset in order to predict heart attacks

 Authors: Julie Favre, Louis Duval, Mathieu 
"""

### LOAD DATA
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset")

### EXTRACT LABELS
labels = np.genfromtxt("dataset/x_test.csv", delimiter=",", dtype = str, max_rows=1)
labels = np.delete(labels,0) # delete the label 'id' as we dont have it in x_train and x_test

### CHOOSE FEATURES TO KEEP
# name of label, exception values, wheter its binary or linear
label_list = [("MSCODE", [], True),
            ("_HCVU651", [9], True),
            ("_RFHYPE5", [9], True),
            ("_RFCHOL", [9], True),
            ("_RACE",[9], True),
            ("_BMI5",[], False),
            ("_EDUCAG",[9], True),
            ("_INCOMG",[9], True),
            ("_DRNKWEK",[99900], False),
            ("_SMOKER3",[9], True),
            ("_FRUTSUM",[], False),
            ("_VEGESUM",[], False),
            ("PA1MIN_",[], False),
            #("GENHLTH",[7,9], False),
            #("CHECKUP1",[7,9], False),
            #("MENTHLTH",[88, 77, 99], False),
            #("BPHIGH4",[7,9], True),
            #("BPMEDS",[7,9], True),
            #("TOLDHI2",[7,9], True),
            ("CHCOCNCR",[7,8,9], True),
            ("DIABETE3",[7,8,9], True),
            #("SEX",[], True),
            #("QLACTLM2",[7,9], True),
            #("AVEDRNK2",[77, 99], False),
            #("EXERANY2",[7,9], True),
            #("SHINGLE2", [7,9], True),
            #("LMTJOIN3", [7,9], True),
            #("CVDASPRN", [7,9], True)
            ]

### CLEAN DATASETS
cleaned_x_train = clean_data(labels, label_list, x_train)
cleaned_x_test = clean_data(labels, label_list, x_test)

### CHOOSE HYPERPARAMETERS
lambda_ = 0.0001
degree = 25


### TRAIN
poly = build_poly(cleaned_x_train,degree)
w , loss = ridge_regression(y_train,poly,lambda_)

### PREDICT LABELS
yp = predict_labels_mse(w,cleaned_x_test)

### WRITE OUTPUTS
create_csv_submission(test_ids,yp,"outputs/ridge_degre25_200features.csv")