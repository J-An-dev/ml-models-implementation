import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tabulate import tabulate


X = np.genfromtxt('data/X.csv', delimiter=',')
y = np.genfromtxt('data/y.csv')

## Utilize sklearn KFlod function to randonmly split dataset
kf = KFold(n_splits=10, shuffle=True, random_state=10)
kf.get_n_splits(X)

i = 9
X_train={}
X_test={}
y_train={}
y_test={}

for train_index, test_index in kf.split(X):
    X_train[i], X_test[i] = X[train_index], X[test_index]
    y_train[i], y_test[i] = y[train_index], y[test_index]
    i = i - 1

## define Navis Bayers model and calculate the prior of y and MLE
def Naive_Bayes(X_train,y_train):
    count = np.count_nonzero(y_train==1)
    pai_1 = count/y_train.shape[0]
    pai_0 = 1-pai_1        #calculate the parameter
    lambda_1 = (1+np.sum(X_train * ((y_train==1)[:,np.newaxis]),axis = 0))/(1+np.sum(y_train==1))
    lambda_0 = (1+np.sum(X_train * ((y_train==0)[:,np.newaxis]),axis = 0))/(1+np.sum(y_train==0))
    lambda_1 = lambda_1[:,np.newaxis]
    lambda_0 = lambda_0[:,np.newaxis]
    return pai_1, pai_0, lambda_1, lambda_0


## calculate MLE
def poission_MLE(feature_col, y_train):
    possible_labels = np.unique(y_train)
    output = []
    for label in possible_labels:
        if int(label) == 0:
            output.append(float(1 + np.sum(feature_col * np.sign(1-y_train))) / (1 + np.sum(np.sign(1 - y_train))))
        else:
            output.append(float(1 + np.sum(feature_col * y_train)) / (1 + np.sum(np.sign(y_train))))
    return output


## calculate poission parameters
def get_MLE_features(X_train, y_train):
    MLE_vector = []
    for feature_col in range(0, X_train.shape[1]):
        MLE_vector.append(poission_MLE(X_train[:, feature_col], y_train))
    return MLE_vector


## plot poission parameters using stem()
def plot_poission_paramters():
    poission_MLE_parameters = {}
    for i in range(0, 10):
        poission_MLE_parameters[i] = get_MLE_features(X_train[i], y_train[i])
        
    average_lambda_parameters = []
    for j in range (0, 54):
        average = [0, 0]
        for i in range (0, 10):
            average = np.sum([average, poission_MLE_parameters[i][j]], axis = 0)
        average = [x / 10 for x in average]
        average_lambda_parameters.append(average)
    class_zero, class_one = zip(*average_lambda_parameters)

    f, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
    markerline, stemlines, baseline = ax1.stem(range(1, 55), class_zero, '-.')
    plt.setp(baseline, 'color', 'c', 'linewidth', 2)
    plt.setp(markerline, 'markerfacecolor', 'b')
    plt.setp(stemlines, 'color', 'b')
    ax1.set_xlabel("Dimensions")
    ax1.set_ylabel("Poission Parameters")
    markerline, stemlines, baseline = ax1.stem(range(1, 55), class_one, '-')
    plt.setp(baseline, 'color', 'c', 'linewidth', 2)
    plt.setp(markerline, 'markerfacecolor', 'y')
    plt.setp(stemlines, 'color', 'y','alpha', 0.5)
    plt.legend(["Class 0", "Class 1"], loc='best', numpoints=2)
    plt.title("Plot for Poission Parameters when y = 0 (Blue) and y = 1 (Yellow)")
    plt.show()
    
    print('The Poission parameters for the 16th dimension are:\nClass 0: %s, Class 1: %s \n' \
      % (average_lambda_parameters[15][0], average_lambda_parameters[15][1]))
    print('The Poission parameters for the 52th dimension are:\nClass 0: %s, Class 1: %s' \
      % (average_lambda_parameters[51][0], average_lambda_parameters[51][1]))


## cross validation for Naive Bayes Classifier
def Cross_Validation_NB():
    c_00, c_01, c_10, c_11 = 0, 0, 0, 0
    lambda_0_sum = np.ones(54)
    lambda_1_sum = np.ones(54)

    for i in range(0, 10):    
        pai_1, pai_0, lambda_1, lambda_0 =  Naive_Bayes(X_train[i], y_train[i])
        lambda_0_sum += np.squeeze(lambda_0)
        lambda_1_sum += np.squeeze(lambda_1)

        #prediction
        lgp_0 = np.log(pai_0) - np.sum(lambda_0) + np.sum(np.dot(X_test[i],np.log(lambda_0)),axis = 1)
        lgp_1 = np.log(pai_1) - np.sum(lambda_1) + np.sum(np.dot(X_test[i],np.log(lambda_1)),axis = 1)
        lgp = np.hstack((lgp_0.reshape(460,1), lgp_1.reshape(460,1)))
        
        #find the class with larger probability
        pred = np.argmax(lgp, axis = 1)

        c_00 += np.sum((pred == 0) * (y_test[i] == 0))
        c_01 += np.sum((pred == 0) * (y_test[i] == 1))
        c_10 += np.sum((pred == 1) * (y_test[i] == 0))
        c_11 += np.sum((pred == 1) * (y_test[i] == 1))
        
    #print the table with prediction results
    table = [['y=0', c_00, c_01],
             ['y=1', c_10, c_11]]
    headers = [' ', "y'=0", "y'=1"]
    acc = (c_00+ c_11)/ (c_00+ c_01+ c_10+c_11)

    print(tabulate(table, headers,tablefmt="fancy_grid"))
    print("prediction accuracy = ", acc)


Cross_Validation_NB()
plot_poission_paramters()