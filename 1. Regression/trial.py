from random import seed
from random import randrange
from csv import reader
from math import sqrt
from operator import add
import numpy as np
import matplotlib.pyplot as plt
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
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
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
    yhat = coefficients[0]
    for i in range(len(row)-2):
        yhat += coefficients[i + 1] * row[i]
    return yhat

# Estimate linear regression coefficients using stochastic gradient descent


def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0])-1)]
    # print("hi`",coef)

    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-2):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
            # print(l_rate, n_epoch, error)
    # print("hi2",coef)
    return coef

# Linear Regression Algorithm With Stochastic Gradient Descent


def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    # global coef_list
    global error_list
    coef = coefficients_sgd(train, l_rate, n_epoch)
    # coef_list.append(coef)
    for row in test:
        yhat = predict(row, coef)
        predictions.append(yhat)
        # error_list.append(yhat - row[-1])
    return predictions, coef


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
    predictions = list()
    coef = np.matmul(np.matmul(np.linalg.inv(
        lambd*np.identity(len(A[0]))+np.matmul(A.T, A)), A.T), y)
    # print(coef)
    for row in A:
        yhat = predict(row, coef)
        predictions.append(yhat)
    sum_error = 0.0
    for i in range(len(y)):
        prediction_error = predictions[i] - y[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(y))
    return sqrt(mean_error)

def scale(X, x_min=-1, x_max=1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

# Linear Regression on wine quality dataset
seed(1)
# load and prepare data

filename = 'Gaussian_noise.csv'
filename20 = 'part1trial.csv'
dataset = load_csv(filename20)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# normalizefilename = 'Gaussian_noise.csv'
# minmax = dataset_minmax(dataset)
# normalize_dataset(dataset, minmax)
n_folds = 5
l_rate = 1e-2
n_epoch = 50
min_rmse = 100
min_deg = 30
min_rmse_mp = 100
min_deg_mp = 30
coef_list = []
rmse_list = []
error_list = []
meta_el = []

for n in range(1, 30):
    basis_data = basis(dataset, n)
    basis_data = scale(np.array(basis_data))
    # evahat = predict(row, coef)luate algorithm
    scores = evaluate_algorithm(
        basis_data, linear_regression_sgd, n_folds, l_rate, n_epoch)
    meta_el.append(error_list)
    error_list = []
    rmse = (sum(scores)/float(len(scores)))
    rmse_list.append(rmse)
    rmse_mp = Moore_Penrose(
        np.array(basis_data)[:, :-1], np.array(basis_data)[:, -1])

    rmse_mp_regu = Moore_Penrose(
        np.array(basis_data)[:, :-1], np.array(basis_data)[:, -1], 10)

    print('----------------Degree: %2d' % n)
    print('Scores: %s' % scores)
    print('Mean RMSE: %.3f' % rmse)
    print('Mean RMSE-MP: %.3f' % rmse_mp)
    print('Mean RMSE-MP-Regularised: %.3f' % rmse_mp_regu)

    if(rmse < min_rmse):
        min_rmse = rmse
        min_deg = n

    if(rmse_mp < min_rmse_mp):
        min_rmse_mp = rmse_mp
        min_deg_mp = n

print('*********** RESULTS ***************')

print('Min Degree: %2d' % min_deg)
print('Min RMSE: %.3f' % min_rmse)
print(coef_list[min_deg-1])

print('Min Moore-Penrose RMSE: %.3f' % min_rmse_mp)
print('Min Degree: %2d' % min_deg_mp)

# plt.plot(rmse_list)
plt.plot(meta_el[min_deg-1])
plt.show()
