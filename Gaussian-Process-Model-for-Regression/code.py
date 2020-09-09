import numpy as np
import matplotlib.pyplot as plt

X_train = np.genfromtxt('data/X_train.csv', delimiter=',')
X_test = np.genfromtxt('data/X_test.csv', delimiter=',')
y_train = np.genfromtxt('data/y_train.csv')
y_test = np.genfromtxt('data/y_test.csv')

#################
# Problem 1 & 2 #
#################

## data standardization and drop the offset column in X_train and X_test
X_train = np.delete(X_train, np.s_[6:7], axis=1)
X_test = np.delete(X_test, np.s_[6:7], axis=1)

X_train_s = (X_train - np.mean(X_train, axis=0))/np.std(X_train, axis=0)
X_test_s = (X_test - np.mean(X_train, axis=0))/np.std(X_train, axis=0)

y_train_s = y_train - np.mean(y_train)
y_test_s = y_test - np.mean(y_train)


## define RBF kernel for high dimension vector
def RBFKernel(x1, x2, b):
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return np.exp(- 1 / b * sqdist)


## calculate normal distribution parameters for predicted y
def conditional(x_new, x, y, b, sigma2):
    
    KxD = RBFKernel(x_new, x, b)
    Kn = RBFKernel(x, x, b)
    Kxx = RBFKernel(x_new, x_new, b)
 
    mu = np.linalg.inv(sigma2 * np.eye(len(Kn)) + Kn).dot(KxD.T).T.dot(y)
    Sigma = sigma2 + Kxx - KxD.dot(np.linalg.inv(sigma2 * np.eye(len(Kn))).dot(KxD.T))
 
    return (mu.squeeze(), Sigma.squeeze())


## predict y
def prediction (mu, Sigma):
    y_pred = np.random.multivariate_normal(mu, Sigma, 100)
    y_pred_mean = np.mean(y_pred, axis=0)
    return y_pred_mean


## calculate RMSE value
def RMSEvalue (y_pred, y_test):
    RMSE = np.sqrt(np.sum(np.square(y_test - y_pred))/len(y_test))
    return RMSE


b = [5,7,9,11,13,15]
sigma2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

RMSE = []
for i in range(0, len(b)):
    for j in range(0, len(sigma2)):
        mu, Sigma = conditional(X_test_s, X_train_s, y_train_s, b[i], sigma2[j])
        y_pred = prediction(mu, Sigma)
        rmse = RMSEvalue (mu, y_test_s)
        RMSE.append(rmse)

MIN = min(RMSE)
for i in range(0, 60):
    if RMSE[i] == min(RMSE):
        n = (i+1)%10
        m = int(((i+1) - n)/10)+1
        print('When b = %s, sigma2 = %s gives the smallest RMSE value: %s' % (str(b[m-1]), str(sigma2[n-1]), MIN))


#############
# Problem 3 #
#############

X_train_4 = X_train_s[:,3]
X_test_4 = X_test_s[:,3]

## define RBF kernel for 1D vector
def Kernel(x1, x2, b):
    return np.exp( - 1/b * np.subtract.outer(x1, x2)**2)

## calculate normal distribution parameters for predicted y
def Conditional(x_new, x, y, b, sigma2):
    
    KxD = Kernel(x_new, x, b)
    Kn = Kernel(x, x, b)
    Kxx = Kernel(x_new, x_new, b)
 
    mu = np.linalg.inv(sigma2 * np.eye(len(Kn)) + Kn).dot(KxD.T).T.dot(y)
    Sigma = sigma2 + Kxx - KxD.dot(np.linalg.inv(sigma2 * np.eye(len(Kn))).dot(KxD.T))
 
    return (mu.squeeze(), Sigma.squeeze())


## predict y
def Prediction (mu, Sigma):
    y_pred = np.random.multivariate_normal(mu, Sigma, 1000)
    y_pred_mean = np.mean(y_pred, axis=0)
    return y_pred_mean


mu, Sigma = Conditional(X_train_4, X_train_4, y_train_s, b=5, sigma2=2)
train_pred = np.hstack((X_train_4.reshape(-1,1), mu.reshape(-1,1)))
train_pred = train_pred[train_pred[:,0].argsort()]
X_train_4_s, pred = zip(*train_pred)


plt.figure(figsize=(15,10))
# plt.scatter(X_test_4, mu, marker='|', color='b')
plt.plot(X_train_4_s, pred, color='b', label='Prediction')
plt.scatter(X_train_4, y_train_s, s=80, marker='.', color='r', label='Ground Truth')

plt.xlabel("X[4]: Car Weight")
plt.ylabel("y")

plt.legend()
plt.show()