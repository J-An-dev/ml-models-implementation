import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline ## for Jupyter Notebook to use matplotlib


##############
#   Part 1   #
##############

# Read data
X_train = np.genfromtxt('data/X_train.csv', delimiter=',')
X_test = np.genfromtxt('data/X_test.csv', delimiter=',')
y_train = np.genfromtxt('data/y_train.csv', delimiter=',')
y_test = np.genfromtxt('data/y_test.csv', delimiter=',')


# define ridge regression computation
def RidgeRegression(X, y, alpha):
    wRR = []
    df = []
    I = np.identity(X.shape[1])
    
    for i in range(0, len(alpha), 1):
        WRR = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + alpha[i] * I), X.T), y)
        DF = np.trace(np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X) + alpha[i] * I)), X.T))
        wRR.append(WRR)
        df.append(DF)
    
    wRR = np.asarray(wRR)
    df = np.asarray(df)
    return wRR, df

# define df figure
def dfplot(wRR, df):
    plt.figure(figsize=(8, 8))
    labels = ["Dimension 1", "Dimension 2", "Dimension 3", "Dimension 4", "Dimension 5", "Dimension 6", "Dimension 7"]
    colors = ['#ff7f00','#ffff33','#a65628','#e41a1c','#377eb8','#4daf4a','#984ea3']
    
    for i in range(0, wRR.shape[1]):
        plt.plot(df, wRR[:, i], color = colors[i])
        plt.scatter(df, wRR[:, i], color = colors[i], label = labels[i])
    
    plt.xlabel("df($\lambda$)")
    plt.ylabel("$w_{RR}$")
    plt.legend()
    plt.show()

# compute wRR and df, then draw  df figure
wRR, df = RidgeRegression(X_train, y_train, alpha = np.arange(0,5001,1).tolist())
dfplot(wRR, df)



# define RMSE value computation
def RMSEvalue(X_test, y_test, wRR, alpha_max):
    RMSE = []
    for i in range(0, alpha_max+1):
        y_pred = np.dot(X_test, wRR[i])
        rmse = np.sqrt(np.sum(np.square(y_test - y_pred))/len(y_test))
        RMSE.append(rmse)
    return RMSE

# define RMSE figure
def RMSEplot(RMSE):
    plt.figure(figsize=(8, 8))
    plt.plot(range(len(RMSE)), RMSE)
    plt.scatter(range(len(RMSE)), RMSE)
    plt.xlabel("$\lambda$")
    plt.ylabel("RMSE")
    plt.show()

# compute RMSE and draw figure
RMSE = RMSEvalue(X_test, y_test, wRR, alpha_max=50)
RMSEplot(RMSE)






##############
#   Part 2   #
##############

# read data
X_train = np.genfromtxt('hw1-data/X_train.csv', delimiter=',')
X_test = np.genfromtxt('hw1-data/X_test.csv', delimiter=',')
y_train = np.genfromtxt('hw1-data/y_train.csv', delimiter=',')
y_test = np.genfromtxt('hw1-data/y_test.csv', delimiter=',')


# git rid of the offset dimension
X_train = np.delete(X_train, np.s_[6:7], axis=1)
X_test = np.delete(X_test, np.s_[6:7], axis=1)


# data standarization processing
X_train_s = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
X_test_s = (X_test - np.mean(X_train, axis=0))/np.std(X_train, axis=0)

y_train_s = y_train - np.mean(y_train)
y_test_s = y_test - np.mean(y_train)


# data dimension expanding
X_train_2 = np.power(X_train[:, 0:6], 2)
X_test_2 = np.power(X_test[:, 0:6], 2)

X_test_3 = np.power(X_test[:, 0:6], 3)
X_train_3 = np.power(X_train[:, 0:6], 3)


# data standarization processing
X_train_2s = (X_train_2 - np.mean(X_train_2, axis=0))/np.std(X_train_2, axis=0)
X_train_3s = (X_train_3 - np.mean(X_train_3, axis=0))/np.std(X_train_3, axis=0)

X_test_2s = (X_test_2 - np.mean(X_train_2, axis=0))/np.std(X_train_2, axis=0)
X_test_3s = (X_test_3 - np.mean(X_train_3, axis=0))/np.std(X_train_3, axis=0)

X_train_2S = np.hstack((X_train_s, X_train_2s))
X_train_3S = np.hstack((X_train_s, X_train_2s, X_train_3s))

X_test_2S = np.hstack((X_test_s, X_test_2s))
X_test_3S = np.hstack((X_test_s, X_test_2s, X_test_3s))


# compute wRR and RMSE
wRR1, df1 = RidgeRegression(X_train_s, y_train_s, alpha = np.arange(0,101,1).tolist())
wRR2, df2 = RidgeRegression(X_train_2S, y_train_s, alpha = np.arange(0,101,1).tolist())
wRR3, df3 = RidgeRegression(X_train_3S, y_train_s, alpha = np.arange(0,101,1).tolist())

RMSE1 = RMSEvalue(X_test_s, y_test_s, wRR1, alpha_max=100)
RMSE2 = RMSEvalue(X_test_2S, y_test_s, wRR2, alpha_max=100)
RMSE3 = RMSEvalue(X_test_3S, y_test_s, wRR3, alpha_max=100)


# draw RMSE figure
plt.figure(figsize=(8,8))
colors = ['#ff7f00','#ffff33','#a65628']
plt.plot(range(101), RMSE1, color = colors[0])
plt.scatter(range(101), RMSE1, color = colors[0], label='p = 1')
plt.plot(range(101), RMSE2, color = colors[1])
plt.scatter(range(101), RMSE2, color = colors[1], label='p = 2')
plt.plot(range(101), RMSE3, color = colors[2])
plt.scatter(range(101), RMSE3, color = colors[2], label='p = 3')

plt.xlabel("$\lambda$")
plt.ylabel("RMSE")

plt.legend()
plt.show()


# compute minimum vlaues for RMSE
print("min RMSE when p = 2: ",min(RMSE2))
print("min RMSE when p = 3: ",min(RMSE3))