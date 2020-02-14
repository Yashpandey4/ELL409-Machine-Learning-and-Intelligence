# Linear Regression With Stochastic Gradient Descent for Wine Quality
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from operator import add
import numpy as np
import matplotlib.pyplot as plt
import random
# Load a CSV file


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Find the min and max values for each column


def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])-1):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate root mean squared error


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

# Evaluate an algorithm using a cross validation split


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    global coef_list
    coef_sum = [0.0 for i in range(len(dataset[0])-1)]
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted, coef = algorithm(train_set, test_set, *args)
        coef_sum = list(map(add, coef_sum, coef))
        actual = [row[-1] for row in fold]
        rmse = rmse_metric(actual, predicted)
        scores.append(rmse)
    coef_sum = [x / n_folds for x in coef_sum]
    # print('******************', coef_sum)
    coef_list.append(coef_sum)
    return scores

# Make a prediction with coefficients


def predict(row, coefficients):
    yhat = 0
    for i in range(len(row)-1):
        yhat += coefficients[i] * row[i]
    return yhat


def predict2(row, coefficients):
    yhat = 0
    for i in range(len(row)):
        yhat += coefficients[i] * row[i]
    return yhat

# Estimate linear regression coefficients using stochastic gradient descent


def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    # print("hi`",coef)
    global error_list
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            error_list.append(error)
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
            # print(l_rate, n_epoch, error)
    # print("hi2",coef)
    return coef

# Linear Regression Algorithm With Stochastic Gradient Descent

def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    # global coef_list
    coef = coefficients_sgd(train, l_rate, n_epoch)
    # coef_list.append(coef)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
    return predictions, coef

# Batch Gradient Descent
def GradientDescent(A, test, l_rate = 0.25, n_epoch = 50, bsize = 10, graph = False):
    global error_list
    predictions = list()
    if(bsize<1):
        bsize=1
    elif(bsize>A.shape[0]):
        bsize=A.shape[0]
    if(graph):
        error_list = np.array([])
    w = np.zeros(A.shape[1],1)
    for i in range(n_epoch):   # Epochs
        j = random.randint(0, A.shape[0]-bsize)
        train = A[j:j+bsize,:]
        for row in train:
            yhat = predict(row, w)
            error = yhat - row[-1]
            if(graph):
                error_list.append(error)
            for i in range(j,j+bsize-1):
                if(j==0):
                    w[0] = w[0] - l_rate * error
                else:
                    w[i + 1] = w[i + 1] - l_rate * error * row[i]
    for row in test:
        yhat = predict(row, w)
        predictions.append(yhat)
    if(graph):
        plt.plot(range(n_epoch, error_list))
        plt.show()
    return predictions, w

def basis(dataset, deg):
    ans = []
    for row in dataset:
        temp = []
        for i in range(deg+1):
            temp.append(row[0] ** i)
        temp.append(row[1])
        ans.append(temp)
    return ans


def Moore_Penrose(A, y, lambd=0):
        # print("hi", y)
    global mp_coef, mp_regu_coef
    predictions = list()
    coef = np.matmul(np.matmul(np.linalg.inv(
        lambd*np.identity(len(A[0]))+np.matmul(A.T, A)), A.T), y)
    if(lambd == 0):
        mp_coef.append(coef)
    else:
        mp_regu_coef.append(coef)
    for row in A:
        yhat = predict2(row, coef)
        predictions.append(yhat)
    sum_error = 0.0
    for i in range(len(y)):
        prediction_error = predictions[i] - y[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(y))
    return sqrt(mean_error)


# Linear Regression with random seed for reproducibility
seed(1)
# load and prepare data

filename = 'Gaussian_noise.csv'
filename20 = 'part1trial.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# normalizefilename = 'Gaussian_noise.csv'
# plt.scatter(dataset[0], dataset[1])
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

# HYPERPARAMETERS
n_folds = 5
l_rate = 0.25
lambd = 1e-9
n_epoch = 50
bsize = 10

min_rmse = 100
min_deg = 30
min_rmse_batch = 100
min_deg_batch = 30
min_rmse_mp = 100
min_deg_mp = 30
min_rmse_mp_regu = 100
min_deg_mp_regu = 30

coef_list = []
rmse_list = []

error_list = []
meta_el_sgd = []
meta_el_batch = []

mp_coef = []
mp_regu_coef = []


for n in range(1, 25):
    basis_data = basis(dataset, n)
    # evaluate algorithm
    scores = evaluate_algorithm(
        basis_data, linear_regression_sgd, n_folds, l_rate, n_epoch)
    meta_el_sgd.append(error_list)
    error_list = []
    rmse = (sum(scores)/float(len(scores)))

    scores_batch = evaluate_algorithm(
        basis_data, GradientDescent, n_folds, l_rate, n_epoch, bsize, True)
    meta_el_batch.append(error_list)
    error_list = []
    rmse_batch = (sum(scores_batch)/float(len(scores_batch)))

    rmse_mp = Moore_Penrose(
        np.array(basis_data)[:, :-1], np.array(basis_data)[:, -1])

    rmse_mp_regu = Moore_Penrose(
        np.array(basis_data)[:, :-1], np.array(basis_data)[:, -1], lambd)

    rmse_list.append(rmse_mp_regu)

    print('----------------Degree: %2d' % n)
    print('Scores: %s' % scores)
    print('Mean RMSE: %.3f' % rmse)
    print('Mean RMSE-MP: %.3f' % rmse_mp)
    print('Mean RMSE-MP-Regularised: %.3f' % rmse_mp_regu)

    if(rmse < min_rmse):
        min_rmse = rmse
        min_deg = n

    if(rmse_batch < min_rmse_batch):
        min_rmse_batch = rmse_batch
        min_deg_batch = n

    if(rmse_mp < min_rmse_mp):
        min_rmse_mp = rmse_mp
        min_deg_mp = n

    if(rmse_mp_regu < min_rmse_mp_regu):
        min_rmse_mp_regu = rmse_mp_regu
        min_deg_mp_regu = n

print('*********** RESULTS ***************')

print('Min Linear Regression RMSE: %.3f' % min_rmse)
print('Min Degree: %2d' % min_deg)
print(coef_list[min_deg-1])

print('Min GD Batch RMSE: %.3f' % min_rmse_batch)
print('Min Degree: %2d' % min_deg_batch)
print(coef_list[min_deg_batch-1])

print('Min Moore-Penrose RMSE: %.3f' % min_rmse_mp)
print('Min Degree: %2d' % min_deg_mp)
print(mp_coef[min_deg_mp-1])

print('Min Moore-Penrose RMSE with Regularisation: %.3f' % min_rmse_mp_regu)
print('Min Degree: %2d' % min_deg_mp_regu)
print(mp_regu_coef[min_deg_mp_regu-1])

# p = np.poly1d(coef_list[min_deg-1][::-1])
# x = np.arange(-20,20)
# y = p(x)
# # plt.plot(x, y)
# # plt.scatter(dataset[0], dataset[1])
# plt.plot(rmse_list)
# plt.show()
plt.plot(rmse_list)
plt.show()
print(rmse_list)
