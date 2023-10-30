import numpy
from helpers import *
from implementations import *

"""
Training and testing our model on the Behavioral Risk Factor Surveillance
 System dataset in order to predict heart attacks

 Authors: Julie Favre, Louis Duval, Mathieu Ferrey
"""

### LOAD DATA
print("loading data")
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset")

### EXTRACT LABELS
labels = np.genfromtxt("dataset/x_test.csv", delimiter=",", dtype = str, max_rows=1)
labels = np.delete(labels,0) # delete the label 'id' as we dont have it in x_train and x_test

### CHOOSE FEATURES TO KEEP
# name of label, exception values, wheter its binary or linear
label_list = [("MSCODE", [], True, None),
            ("_HCVU651", [9], True, None),
            ("_RFHYPE5", [9], True, None),
            ("_RFCHOL", [9], True, None),
            ("_RACE",[9], True, None),
            ("_BMI5",[], False, None),
            ("_EDUCAG",[9], True, None),
            ("_INCOMG",[9], True, None),
            ("_DRNKWEK",[99900], False, None),
            ("_SMOKER3",[9], True, None),
            ("_FRUTSUM",[], False, None),
            ("_VEGESUM",[], False, None),
            ("PA1MIN_",[], False, None),
            ("GENHLTH",[7,9], False, None),
            ("CHECKUP1",[7,9], False, None),
            ("MENTHLTH",[77, 99], False, 88), #88 -> 0 
            ("BPHIGH4",[7,9], True, None),
            ("BPMEDS",[7,9], True, None),
            ("TOLDHI2",[7,9], True, None),
            ("CHCOCNCR",[7,8,9], True, None),
            ("DIABETE3",[7,8,9], True, None),
            ("SEX",[], True, None),
            ("QLACTLM2",[7,9], True, None),
            ("AVEDRNK2",[77, 99], False, None),
            ("EXERANY2",[7,9], True, None),
            ("SHINGLE2", [7,9], True, None),
            ("LMTJOIN3", [7,9], True, None),
            ("CVDASPRN", [7,9], True, None),
            ("PHYSHLTH",[77,99], False, 88), #88 -> 0 
            ("POORHLTH",[77,99], False, 88), # 88 -> 0  
            ("HLTHPLN1",[7,9], True, None),
            ("BLOODCHO", [7,9], True, None),
            ("CHOLCHK",[7,9],True, None),
            ("CVDSTRK3", [7,9], True, None),
            ("ASTHMA3", [7,9], True, None),
            ("ASTHNOW", [7,9], True, None),
            ("CHCSCNCR", [7,9], True, None),
            ("CHCCOPD1",[7,9], True, None),
            ("HAVARTH3", [7,9], True, None),
            ("ADDEPEV2", [7,9], True, None),
            ("CHCKIDNY", [7,9], True, None),
            ("DIABAGE2", [98,99], False, None),
            ("MARITAL", [9], True, None),
            ("RENTHOM1", [7,9], True, None),
            ("EMPLOY1", [9], True, None),
            ("CHILDREN", [99], False, 88), # 88 -> 0 
            ("INCOME2", [77,99], True, None),
            ("INTERNET", [7,9], True, None),
            ("QLACTLM2", [7,9], True, None),
            ("USEEQUIP", [7,9], True, None),
            ("BLIND", [7,9], True, None),
            ("DECIDE", [7,9], True, None),
            ("DIFFWALK", [7,9], True, None),
            ("DIFFDRES", [7,9], True, None),
            ("DIFFALON", [7,9], True, None),
            ("SMOKE100", [7,9], True, None),
            ("SMOKDAY2", [7,9], True, None),
            ("STOPSMK2", [7,9], True, None),
            ("USENOW3", [7,9], True, None), 
            ("DRNK3GE5", [77,99],False, 88)  ## 88 -> 0 
            ]


### CLEAN DATASETS
cleaned_x_train = clean_data(labels, label_list, x_train)
cleaned_x_test = clean_data(labels, label_list, x_test)

### CHOOSE HYPERPARAMETERS
lambda_ = 0.1
degree = 15


### TRAIN
poly_train = build_poly(cleaned_x_train,degree)
w , loss = ridge_regression(y_train,poly_train,lambda_)

### PREDICT LABELS
poly_test = build_poly(cleaned_x_test, degree)
yp = predict_labels_mse(w,poly_test)

### WRITE OUTPUTS
path = "outputs/ridge_degre15_lambda1e-1.csv"
create_csv_submission(test_ids,yp,path)
print("The output csv file is at:")
print(path)