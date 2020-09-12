import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

nyt_doc = pd.read_csv('./data/nyt_data.txt',sep='\n',header=None)
X = np.zeros((3012,8447))
doc_index = 0

for i in range(0, nyt_doc.shape[0]):
    row = nyt_doc.iloc[i,:].tolist()[0].split(',')
    for item in row:
        word = item.split(':')
        X[int(word[0])-1, doc_index] = int(word[1])

    doc_index += 1

# Implement Matrix Factorization using divergence objective
def NMF(W, H, num_iter):

    objective = []
    
    for i in range(0,num_iter):
        #update H
        purple = X / (np.matmul(W,H)+1e-16)
        W_norm = W.T / np.sum(W.T, axis=1).reshape(-1,1) #normalize rows
        H = np.multiply(H, np.matmul(W_norm, purple))
        
        #update W
        purple = X / (np.matmul(W,H)+1e-16)  #recalculate purple
        H_norm = H.T / np.sum(H.T, axis=0) #normalize columns
        W = np.multiply(W, np.matmul(purple, H_norm))

        #update objective
        WH = np.matmul(W,H)
        obj = np.sum(np.multiply(np.log(1/(WH+1e-16)),X) + WH)
        objective.append(obj)
    
    return W, H, objective

init_W = np.random.uniform(1,2,(3012,25))
init_H = np.random.uniform(1,2,(25,8447))
W_NMF, H_NMF, obj_lst = NMF(init_W, init_H,100)


# problem 1
plt.figure(figsize=(10,10))
plt.plot(np.arange(1,101), obj_lst)
plt.title('Divergence objective of NMF by iterations')
plt.xlabel('Iterations')
plt.ylabel('Divergence objective')


# problem 2
W_NMF_norm = W_NMF / (np.sum(W_NMF, axis=0).reshape(1,-1))
word = pd.read_csv('./data/nyt_vocab.dat', header=None)
word_index = np.arange(0, 3012)
dict_word = {'word_index': word_index, 'word': word.iloc[:,0].tolist()}
word = pd.DataFrame(dict_word)
word.columns = ['word_index', 'word']

df_W = pd.DataFrame(W_NMF_norm)
table = []

for i in range(df_W.shape[1]):
    word_index = np.arange(0, 3012)
    d = {'word_index':word_index, 'weight':df_W.iloc[:,i].tolist()}
    df = pd.DataFrame(d)
    df = df.sort_values(by='weight', ascending=False).iloc[0:10,:]
    df2 = pd.concat([df, word], axis=1, join='inner').head(25)
    df2 = df2.drop(columns=['word_index'])
    table.append(df2) 


for i in range(25):
    print('Topic_%d'%(i+1))
    print(table[i].to_string(index=False))
    print('\n')