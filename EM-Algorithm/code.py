import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def EM_GMM(data, K, n_iter):
    """
    Return list that contains change of objective function per iteration.

    :param data: dataset
    :param K: num of clusters
    :param n_iter: number of iterations
    """
    # Initialize all covariance matrices to the empirical covariance of the data being modeled
    init_cov_temp = np.cov(data.transpose())
    init_cov = np.array([init_cov_temp] * K)

    # Randomly initialize the means by sampling from a single multivariate Gaussian where the parameters
    # are the mean and covariance of the data being modeled
    data_mean = np.array(data.describe().loc['mean', :])
    init_mean_all = np.random.multivariate_normal(data_mean, init_cov[0], K)

    # Initialize the mixing weights to be uniform
    pi_weight = np.ones(K) * np.array(1 / K)
    n = data.shape[0]

    # EM Algorithm
    # E Step

    obj_func = []
    for i in range(0, n_iter):
        # Update phi denominator
        phi_denominator = 0
        phi = [0] * K
        for i in range(0, K):
            phi_denominator = multivariate_normal.pdf(data, mean=init_mean_all[i], cov=init_cov[i],
                                                      allow_singular=True) * pi_weight[i] + phi_denominator
        # E step: update phi
        for i in range(0, K):  # shape:(1631,)
            phi[i] = pi_weight[i] * multivariate_normal.pdf(data, mean=init_mean_all[i], cov=init_cov[i],
                                                            allow_singular=True) / phi_denominator
            # M Step
        nk = np.zeros(K)
        for i in range(0, K):
            nk[i] = np.sum(phi[i])
            pi_weight[i] = nk[i] / n

        # Update mu
        for i in range(0, K):
            init_mean_all[i] = (1 / nk[i]) * np.matmul(np.matrix(phi[i].reshape(1, -1)), np.matrix(data))

        # Update cov
        x_mu = np.array(data) - init_mean_all[i]  # xi-uk term
        init_cov[i] = np.matmul(np.multiply(phi[i].reshape(-1, 1), x_mu).transpose(), x_mu) / nk[i]

        # Update objective function
        L = np.sum(np.log(phi_denominator))
        obj_func.append(L)

    return obj_func, pi_weight, init_mean_all, init_cov

def implement_GMM_plot(data, K, n_iter, n_times, cls):
    """
    Implement GMM algorithm n times and return a plot.

    :param n_times: n times
    """
    obj_func_all = []
    pi_k_all = []
    init_mean_ALL = []
    init_cov_all = []
    for i in range(0, n_times):
        obj_func, pi_weight, init_mean_all, init_cov = EM_GMM(data, K, n_iter)
        obj_func_all.append(obj_func)
        pi_k_all.append(pi_weight)
        init_mean_ALL.append(init_mean_all)
        init_cov_all.append(init_cov)

    i = 1
    plt.figure(figsize=(10, 10))
    for item in obj_func_all:
        item = item[4:]  # Start to plot at iter 5
        plt.plot(np.arange(5, 31), item, label='run_' + str(i))
        i += 1
        plt.legend()
        plt.title(str(K) + '-GMM Class ' + str(cls) + ' log marginal objective function by iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Log Marginal Objective Function')
    plt.show()
    return obj_func_all, pi_k_all, init_mean_ALL, init_cov_all

def best_run(obj_func_all):
    """
    Return the index of iteration that has the largest log marginal likelihood (best run).
    """
    last = []
    for item in obj_func_all:
        last.append(item[-1])
        MAX = max(last)
        ind = [i for i,j in enumerate(last) if j == MAX]
        max_ind = ind[0]
    return max_ind

def GMM_Bayes_Classifier(data, K, n_iter, n_times):
    """
    Return a list containing prediction result using GMM and Bayes Classifier.

    :param K: K-Gaussian Mixture Model
    """
    obj_func_all_1, pi_weight_1, init_mean_ALL_1, init_cov_all_1 = implement_GMM_plot(X_train_1, K, n_iter, n_times,
                                                                                      cls=1)
    obj_func_all_0, pi_weight_0, init_mean_ALL_0, init_cov_all_0 = implement_GMM_plot(X_train_0, K, n_iter, n_times,
                                                                                      cls=0)

    ind_1 = best_run(obj_func_all_1)
    ind_0 = best_run(obj_func_all_0)

    best_weight_1 = pi_weight_1[ind_1]
    best_mean_1 = init_mean_ALL_1[ind_1]
    best_cov_1 = init_cov_all_1[ind_1]

    best_weight_0 = pi_weight_0[ind_0]
    best_mean_0 = init_mean_ALL_0[ind_0]
    best_cov_0 = init_cov_all_0[ind_0]

    class1 = 0
    for i in range(0, K):
        class1 += multivariate_normal.pdf(data, mean=best_mean_1[i], cov=best_cov_1[i], allow_singular=True) * \
                  best_weight_1[i]

    class0 = 0
    for i in range(0, K):
        class0 += multivariate_normal.pdf(data, mean=best_mean_0[i], cov=best_cov_0[i], allow_singular=True) * \
                  best_weight_0[i]

    class1 = class1.tolist()
    class0 = class0.tolist()

    pred_result = []
    for i in range(0, len(class1)):
        if class1[i] >= class0[i]:
            pred_result.append(1)
        else:
            pred_result.append(0)
    return pred_result

def pred_result(pred_result):
    """
    Return confusion matrix and accuracy.
    """
    true_list = y_test.iloc[:, 0].tolist()
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0, len(true_list)):
        if pred_result[i] == 1 and true_list[i] == 1:
            TP += 1
        elif pred_result[i] == 0 and true_list[i] == 1:
            FN += 1
        elif pred_result[i] == 1 and true_list[i] == 0:
            FP += 1
        elif pred_result[i] == 0 and true_list[i] == 0:
            TN += 1

    data = [('TP:' + str(TP), 'FP:' + str(FP)), ('FN:' + str(FN), 'TN:' + str(TN))]
    df = pd.DataFrame(data)
    df = df.rename({0: 'Predicted Postive', 1: 'Predicted Negative'}, axis='index')
    df = df.rename({0: 'Actual Postive', 1: 'Actual Negative'}, axis='columns')
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(df)
    print('Accuracy:' + str(accuracy))


# Load dataset
X_train = pd.read_csv('./data/Xtrain.csv', header=None)
y_train = pd.read_csv('./data/ytrain.csv', header=None)
X_test = pd.read_csv('./data/Xtest.csv', header=None)
y_test = pd.read_csv('./data/ytest.csv', header=None)

# Split data into two classes
train_set = pd.concat([X_train, y_train], axis=1, sort=False)
train_1 = train_set[train_set.iloc[:, -1] == 1]
train_0 = train_set[train_set.iloc[:, -1] == 0]
X_train_1 = train_1.iloc[:, 0:10]
X_train_0 = train_0.iloc[:, 0:10]
y_train_1 = train_1.iloc[:, -1]
y_train_0 = train_0.iloc[:, -1]

pred_3GMM = GMM_Bayes_Classifier(X_test, K=3, n_iter=30, n_times=10)
pred_result(pred_3GMM)

pred_1GMM = GMM_Bayes_Classifier(X_test, K=1, n_iter=30, n_times=10)
pred_result(pred_1GMM)

pred_2GMM = GMM_Bayes_Classifier(X_test, K=2, n_iter=30, n_times=10)
pred_result(pred_2GMM)

pred_4GMM = GMM_Bayes_Classifier(X_test, K=4, n_iter=30, n_times=10)
pred_result(pred_4GMM)
