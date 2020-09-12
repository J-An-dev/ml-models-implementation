import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

data = pd.read_csv("./data/CFB2019_scores.csv", header = None)
data.columns = ['Team_A_index','Team_A_points', 'Team_B_index', 'Team_B_points']
team_names = pd.read_csv('./data/TeamNames.txt',sep="\n", header=None)

# Construct random walk matrix M, team_index = 1,2,...,769
M = np.zeros((769, 769))

index = pd.DataFrame(np.arange(1,770))
teams = pd.concat([index, team_names], axis=1, sort=False)
teams.columns = ['team_index','team_name']

for i in range(0, data.shape[0]):
    row = data.iloc[i,:]
    denominator = row.Team_A_points + row.Team_B_points
    # if Team A wins
    if row.Team_A_points > row.Team_B_points:
        M[int(row.Team_A_index-1), int(row.Team_A_index-1)] += 1 + row.Team_A_points/denominator
        M[int(row.Team_B_index-1), int(row.Team_B_index-1)] +=     row.Team_B_points/denominator
        M[int(row.Team_A_index-1), int(row.Team_B_index-1)] +=     row.Team_B_points/denominator
        M[int(row.Team_B_index-1), int(row.Team_A_index-1)] += 1 + row.Team_A_points/denominator
    
    # if Team B wins
    elif row.Team_A_points < row.Team_B_points:
        M[int(row.Team_A_index-1), int(row.Team_A_index-1)] +=     row.Team_A_points/denominator
        M[int(row.Team_B_index-1), int(row.Team_B_index-1)] += 1 + row.Team_B_points/denominator
        M[int(row.Team_A_index-1), int(row.Team_B_index-1)] += 1 + row.Team_B_points/denominator
        M[int(row.Team_B_index-1), int(row.Team_A_index-1)] +=     row.Team_A_points/denominator

M_norm = M/np.sum(M, axis=1).reshape(-1,1)


# problem 1
def markov_chain(t):
    w = np.repeat(1/769, 769).reshape(1, -1)
    
    for i in range(t):
        w = np.matmul(w, M_norm)
    
    d = {'index': np.arange(1, 770), 'score': w.tolist()[0]}
    df = pd.DataFrame(d)
    temp = df.sort_values(by=['score'],ascending=False)
    result = pd.concat([temp, teams], axis=1, join='inner').head(25)[['team_name','score']]
    return result

t_10 = markov_chain(10)
print(t_10.to_string(index=False))

t_100 = markov_chain(100)
print(t_100.to_string(index=False))

t_1000 = markov_chain(1000)
print(t_1000.to_string(index=False))

t_10000 = markov_chain(10000)
print(t_10000.to_string(index=False))


# problem 2
eigen_vector = eigs(M_norm.T,1)[1].flatten()
w_inf = eigen_vector / np.sum(eigen_vector)
diff = []
w = np.repeat(1/769, 769)

for i in range(10000):
    w = np.matmul(w, M_norm)
    diff.append(np.sum(np.abs(w - w_inf)))

plt.figure(figsize = (10, 10))
plt.plot(range(1, 10001), diff)
plt.xticks(np.linspace(1, 10000,9))
plt.xlabel('Iterations t')
plt.ylabel('l1-norm between w_t and w_inf')
plt.title('Variation of difference between w_t and w_inf by iterations')
plt.show()





