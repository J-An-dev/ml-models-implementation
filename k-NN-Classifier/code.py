import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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


## define kNN model
def kNN(X_train, y_train, X_test, y_test, k):
    c_00, c_01, c_10, c_11 = 0, 0, 0, 0
    pred = []
    for i in X_test:
        dist = np.sum(np.abs(X_train-i),axis = 1)
        nns = np.argsort(dist)[:k]
        c = np.array([0,0])
        for m in nns:
            if y_train[m] == 0:
                c[0] += 1
            else:
                c[1] += 1
        label_i = np.argmax(c)
        pred.append(label_i)
    pred = np.asarray(pred)
    c_00 += np.sum((pred == 0) * (y_test == 0))
    c_01 += np.sum((pred == 0) * (y_test == 1))
    c_10 += np.sum((pred == 1) * (y_test == 0))
    c_11 += np.sum((pred == 1) * (y_test == 1))
    return(c_00, c_01, c_10, c_11)


## define cross validation for kNN
def Cross_Validation_kNN(k):
    acc_list = []
    for i in range(k):
        c_00_sum, c_01_sum, c_10_sum, c_11_sum = 0, 0, 0, 0

        for j in range(0, 10):
            c_00, c_01, c_10, c_11 = kNN(X_train[j], y_train[j], X_test[j], y_test[j], i+1)
            c_00_sum += c_00
            c_01_sum += c_01
            c_10_sum += c_10
            c_11_sum += c_11
        print(c_00_sum, c_01_sum, c_10_sum, c_11_sum)
        acc = (c_00_sum + c_11_sum)/(c_00_sum + c_01_sum + c_10_sum + c_11_sum)
        acc_list.append(acc)
        acc_array = np.asarray(acc_list)

        print("The accracy of k = " + str(i+1) + " is: " + str(acc))

    print("The k with max accuracy is: ", np.argmax(np.array(acc_list)) + 1)
    k_array = np.arange(1,21)
    plt.figure(figsize=(15, 6))
    plt.plot(k_array, acc_array, '-o')
    plt.title("Plot of the prediction accuracy of KNN Classifier as a function of k (Number of Neighbours)")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()


Cross_Validation_kNN(20)