#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.metrics import f1_score
import cvxopt
import warnings

import os
os.chdir("/home/prats/Final")


# In[2]:


# All constants goes here
cv = 5 # cross validation
kernels = ['linear', 'rbf', 'poly']
warnings.filterwarnings('ignore')


# In[3]:


def crossval_score(clf, X, t, cv = 4):
    score = np.array([])
    n = X.shape[0]
    batch_size = n//cv
    for i in range(cv):
        beg, end = i*batch_size, (i+1)*batch_size
        test_X, test_t = X[beg:end,:], t[beg:end]
        train_X, train_t = np.delete(X, range(beg, end), axis = 0), np.delete(t, range(beg, end))
        clf.fit(train_X, train_t)
        predicted_t = clf.predict(test_X)
        f1 = f1_score(test_t, predicted_t, average='weighted')
        score = np.append(score, f1)
    return score


# In[4]:


def plotVariations(kernel, cr, gr, dr, cvscores, tscores):
    if (kernel == 'linear'):
        # Variation with C only
        plt.plot(cr, cvscores, label='cross-validation')
        plt.plot(cr, tscores, label='training')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.xscale('log')
        plt.title('Accuracy v/s C')
        plt.legend(loc='best')
        plt.show()
    elif (kernel == 'rbf' or kernel == 'sigmoid'):
        # Variation with C and gamma
        gr, cr = np.meshgrid(gr, cr)
        cvscores = cvscores.reshape(cr.shape)
        fig, ax = plt.subplots()
        cs = ax.contourf(cr, gr, cvscores)
        cbar = fig.colorbar(cs)
        plt.xlabel('C')
        plt.ylabel('gamma')
        plt.title('Contour plot for Accuracy v/s C and gamma (cross-validation)')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        # training
        tscores = tscores.reshape(cr.shape)
        fig, ax = plt.subplots()
        cs = ax.contourf(cr, gr, tscores)
        cbar = fig.colorbar(cs)
        plt.xlabel('C')
        plt.ylabel('gamma')
        plt.title('Contour plot for Accuracy v/s C and gamma (training)')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    else:
        # Variation with C, gamma and degree - poly kernals
        gr, cr = np.meshgrid(gr, cr)
        for d in dr:
            cvscores[d] = cvscores[d].reshape(cr.shape)
            fig, ax = plt.subplots()
            cs = ax.contourf(cr, gr, cvscores[d])
            cbar = fig.colorbar(cs)
            plt.xlabel('C')
            plt.ylabel('gamma')
            plt.title('Contour plot for Accuracy v/s C and gamma, degree=%i (cross-validation)'%(d))
            plt.xscale('log')
            plt.yscale('log')
            plt.show()
            # training
            tscores[d] = tscores[d].reshape(cr.shape)
            fig, ax = plt.subplots()
            cs = ax.contourf(cr, gr, tscores[d])
            cbar = fig.colorbar(cs)
            plt.xlabel('C')
            plt.ylabel('gamma')
            plt.title('Contour plot for Accuracy v/s C and gamma, degree=%i (training)'%(d))
            plt.xscale('log')
            plt.yscale('log')
            plt.show()


# In[5]:


def findParameters(kernels, X, t, cr = np.logspace(0, 1, 10), gr = np.logspace(0, 1, 10), dr = range(1, 5), op = False):
    # returns the classifier with the best parameters
    best_score = np.array([-1])
    for kernel in kernels:
        if (kernel == 'poly'):
            cnt=0
            allscores = {}
            tscores = {}
            for d in dr:
                allscores[d] = np.array([])
                tscores[d] = np.array([])
        else:
            allscores = np.array([])
            tscores = np.array([])
        if (op):
            print('Kernel = %s' % (kernel))
        for c in cr:
            g2 = False
            for g in gr:
                if (g2 and kernel == 'linear'):
                    break
                g2 = True
                d2 = False
                for d in dr:
                    if (d2 and kernel != 'poly'):
                        break
                    d2 = True
                    clf = svm.SVC(kernel = kernel, C = c, gamma = g, degree = d)
                    score = crossval_score(clf, X, t, cv = cv)
                    if (kernel == 'poly'):
                        allscores[d] = np.append(allscores[d], score.mean())
                        tscores[d] = np.append(tscores[d], clf.fit(X, t).score(X, t))
                    else:
                        allscores = np.append(allscores, score.mean())
                        tscores = np.append(tscores, clf.fit(X, t).score(X, t))
                    if (op):
                        print('c = %0.4f, g = %0.4f, d = %i, score = %0.4f (+/- %0.4f)'%(c, g, d, score.mean(), 2*score.std()))
                    if (score.mean() > best_score.mean()):
                        best_score = score
                        best_clf = clf
        plotVariations(kernel, cr, gr, dr, allscores, tscores)
    return best_clf.fit(X, t), best_score


# In[6]:


# The classifier using cvxopt
class mysvm:
    def __init__(this, kernel='rbf', C=1.0, gamma=1.0, degree=3, coef0=0.0, threshold=1e-5):
        this.kernel = kernel
        this.C = C
        this.gamma = gamma
        this.degree = degree
        this.coef0 = coef0
        this.threshold = threshold
    
    def buildK(this, X1, X2):
        K = X1.dot(X2.T)
        if (this.kernel == 'poly'):
            K = (this.gamma * K + this.coef0)**this.degree
        elif (this.kernel == 'sigmoid'):
            K = np.tanh(this.gamma * K + this.coef0)
        elif (this.kernel == 'rbf'):
            sq1 = np.diag(X1.dot(X1.T)).reshape((-1, 1))*np.ones((1, X2.shape[0]))
            sq2 = np.diag(X2.dot(X2.T)).reshape((1, -1))*np.ones((X1.shape[0], 1))
            K = 2*K-sq1-sq2
            K = np.exp(this.gamma * K)
        # All other kernels are treated to be linear
        this.K = K
    
    def fit(this, X, t):
        this.N = X.shape[0]
        this.buildK(X, X)
        t = t.reshape((-1, 1)) * 1.0
        P = cvxopt.matrix((t.dot(t.T))*this.K)
        q = cvxopt.matrix(-np.ones((this.N, 1)))
        G = cvxopt.matrix(np.concatenate((-np.eye(this.N), np.eye(this.N))))
        h = cvxopt.matrix(np.concatenate((np.zeros((this.N, 1)), this.C * np.ones((this.N, 1)))))
        A = cvxopt.matrix(t.T)
        b = cvxopt.matrix(0.0)
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        mu = np.array(sol['x']) # The lagrange multipliers
        this.sv_idx = np.where(mu > this.threshold)[0] # indices of all the support vectors
        
        # Extract the support vectors
        this.sx = X[this.sv_idx,:]
        this.st = t[this.sv_idx]
        this.mu = mu[this.sv_idx]
        
        this.b = np.sum(this.st)
        for j in this.sv_idx:
            this.b -= np.sum(this.mu * this.st * (this.K[j, this.sv_idx].reshape((-1, 1))))
        this.b /= this.sv_idx.shape[0]
        
        # Make the classifier
        def predict(X):
            this.buildK(this.sx, X)
            this.y = np.zeros((X.shape[0],))
            for i in range(this.y.shape[0]):
                this.y[i] = np.sum(this.mu * this.st * this.K[:,i].reshape((-1, 1)))+this.b
            return np.sign(this.y)
        
        this.predict = predict    


# In[7]:


# All data is read here
data = np.array(pd.read_csv('2017EE10938.csv', header = None))
X = data[:, 0:-1]
t = data[:, -1].astype('int')
scl_p1 = max(X.max(), -X.min())
X = X/scl_p1

train_data = np.array(pd.read_csv('train_set.csv', header = None))
train_X = train_data[:, 0:-1]
train_t = train_data[:, -1].astype('int')
scl_p2 = max(train_X.max(), -train_X.min())
train_X = train_X/scl_p2

prediction_data = np.array(pd.read_csv('test_set.csv', header = None))
prediction_X = prediction_data[:,:]
prediction_X = prediction_X/scl_p2


# In[8]:


# Part 1 (Binary Classification) using all 25 features
pairs = [(0, 1), (2, 3), (4, 5)]
for i, j in pairs:
    print('Pair: (%i, %i)'%(i, j))
    idx_i = np.where(t == i)[0]
    idx_j = np.where(t == j)[0]
    idx = np.concatenate((idx_i, idx_j))
    np.random.shuffle(idx)
    Xp = X[idx,:]
    tp = t[idx]
    tp = np.where(tp==i, -1, 1)
    for kernel in kernels:
        print('Kernel: %s'%(kernel))
        clf_p, cvscore_p = findParameters([kernel], Xp, tp)
        param_p = clf_p.get_params()
        myclf = mysvm(kernel=kernel, C=param_p['C'], gamma=param_p['gamma'], degree=param_p['degree'], coef0=param_p['coef0'])
        mycvscore = crossval_score(myclf, Xp, tp, cv = cv)
        myclf.fit(Xp, tp)
        print(param_p)
        print('sklearn cross-val score: %0.4f, cvxopt cross-val score: %0.4f'%(cvscore_p.mean(), mycvscore.mean()))
        print('sklearn num of SV: %i, cvxopt num of SV: %i'%(clf_p.support_.shape[0], myclf.sv_idx.shape[0]))
        print('sklearn indices of support vector:', np.sort(clf_p.support_))
        print('cvxopt indices of support vector :', myclf.sv_idx)
    print('\n')


# In[9]:


# Part 1 (Binary Classification) using 10 features
pairs = [(0, 1), (2, 3), (4, 5)]
for i, j in pairs:
    print('Pair: (%i, %i)'%(i, j))
    idx_i = np.where(t == i)[0]
    idx_j = np.where(t == j)[0]
    idx = np.concatenate((idx_i, idx_j))
    np.random.shuffle(idx)
    Xp = X[idx,:10]
    tp = t[idx]
    tp = np.where(tp==i, -1, 1)
    for kernel in kernels:
        print('Kernel: %s'%(kernel))
        clf_p, cvscore_p = findParameters([kernel], Xp, tp)
        param_p = clf_p.get_params()
        myclf = mysvm(kernel=kernel, C=param_p['C'], gamma=param_p['gamma'], degree=param_p['degree'], coef0=param_p['coef0'])
        mycvscore = crossval_score(myclf, Xp, tp, cv = cv)
        myclf.fit(Xp, tp)
        print(param_p)
        print('sklearn cross-val score: %0.4f, cvxopt cross-val score: %0.4f'%(cvscore_p.mean(), mycvscore.mean()))
        print('sklearn num of SV: %i, cvxopt num of SV: %i'%(clf_p.support_.shape[0], myclf.sv_idx.shape[0]))
        print('sklearn indices of support vector:', np.sort(clf_p.support_))
        print('cvxopt indices of support vector :', myclf.sv_idx)
    print('\n')


# In[10]:


# Part 1 (Multiclass Classification) using all 25 features
for kernel in kernels:
    print('Kernel: %s'%(kernel))
    clf_p1, cvscore_p1 = findParameters([kernel], X, t)
    print("Score: %0.4f (+/- %0.4f)" % (cvscore_p1.mean(), 2*cvscore_p1.std()))
    print(clf_p1.get_params())


# In[11]:


# Part 1 (Multiclass Classification) using 10 features
for kernel in kernels:
    print('Kernel: %s'%(kernel))
    clf_p1, cvscore_p1 = findParameters([kernel], X[:,:10], t)
    print("Score: %0.4f (+/- %0.4f)" % (cvscore_p1.mean(), 2*cvscore_p1.std()))
    print(clf_p1.get_params())


# In[12]:


# Part 2 (training)
clf_p2, cvscore_p2 = findParameters(kernels, train_X, train_t)
print("Score: %0.4f (+/- %0.4f)" % (cvscore_p2.mean(), 2*cvscore_p2.std()))
print(clf_p2.get_params())


# In[13]:


# Part 2 (prediction)
prediction_t = clf_p2.predict(prediction_X)
print(prediction_t)
df = pd.DataFrame({'class': prediction_t})
df.to_csv('prediction.csv', index_label = 'id')


# In[14]:


def kagglePar(X, t):
    beg_c, end_c = 1e-2, 1e2
    for i in range(10):
        c1 = (2*beg_c+end_c)/3
        beg_g, end_g = 1e-2, 1e2
        for j in range(10):
            g1 = (2*beg_g+end_g)/3
            clf = svm.SVC(kernel='rbf', C=c1, gamma = g1)
            scoreg1 = crossval_score(clf, X, t)
            g2 = (beg_g+2*end_g)/3
            clf = svm.SVC(kernel='rbf', C=c1, gamma = g2)
            scoreg2 = crossval_score(clf, X, t)
            if scoreg1.mean() < scoreg2.mean():
                beg_g = g1
            else:
                end_g = g2
        g1 = (beg_g+end_g)/2
        clf = svm.SVC(kernel='rbf', C=c1, gamma=g1)
        scorec1 = crossval_score(clf, X, t)
        
        c2 = (2*beg_c+end_c)/3
        beg_g, end_g = 1e-5, 1e5
        for j in range(10):
            g1 = (2*beg_g+end_g)/3
            clf = svm.SVC(kernel='rbf', C=c2, gamma = g1)
            scoreg1 = crossval_score(clf, X, t)
            g2 = (beg_g+2*end_g)/3
            clf = svm.SVC(kernel='rbf', C=c2, gamma = g2)
            scoreg2 = crossval_score(clf, X, t)
            if scoreg1.mean() < scoreg2.mean():
                beg_g = g1
            else:
                end_g = g2
        g2 = (beg_g+end_g)/2
        clf = svm.SVC(kernel='rbf', C=c2, gamma=g2)
        scorec2 = crossval_score(clf, X, t)
        if scorec1.mean() < scorec2.mean():
            beg_c = c1
        else:
            end_c = c2
    
    c = (beg_c+end_c)/2;
    for j in range(10):
        g1 = (2*beg_g+end_g)/3
        clf = svm.SVC(kernel='rbf', C=c, gamma = g1)
        scoreg1 = crossval_score(clf, X, t)
        g2 = (beg_g+2*end_g)/3
        clf = svm.SVC(kernel='rbf', C=c, gamma = g2)
        scoreg2 = crossval_score(clf, X, t)
        if scoreg1.mean() < scoreg2.mean():
            beg_g = g1
        else:
            end_g = g2
    g = (beg_g+end_g)/2
    clf = svm.SVC(kernel='rbf', C=c, gamma=g)
    score = crossval_score(clf, X, t)
    return clf, score


# In[ ]:


# Further Training - Improving score on Kaggle
clf_k, score_k = kagglePar(train_X, train_t)
print("Score: %0.4f (+/- %0.4f)" % (score_k.mean(), 2*score_k.std()))
print(clf_k.get_params())


# In[ ]:




