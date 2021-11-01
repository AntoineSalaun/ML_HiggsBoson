import numpy as np
from Utility import *
from implementation import *

def cross_validation(folds, X, y, initial_w, max_iters, gamma, lambda_, method = 'least_squares_GD'):
    
    # split the data in folds
    X_split = np.array_split(X, folds, axis=1)
    y_split = np.array_split(y, folds)
    
    # Define the arrays to store weights, losses and MSE for the K trains and tests
    weights = np.empty(folds, dtype=object) 
    losses = np.empty(folds, dtype=object) 
    MSE = np.empty(folds, dtype=object) 
    
    # For each fold :
    for i in range(folds):

        # The training will be done on every fold except the running number
        fold_train = list(range(folds))
        del fold_train[i]
    
        # Creating the train set by joining together every other fold
        X_train= np.concatenate([X_split[k] for k in fold_train],axis=1)
        y_train= np.concatenate([y_split[k] for k in fold_train])
    
        # test sets
        X_test= X_split[i]
        y_test= y_split[i]
    
        # The model is trained
        if method == 'least_squares_GD':
            weights[i], losses[i] = least_squares_GD(y_train, X_train.T , initial_w, max_iters, gamma)
        if method == 'least_squares_SGD':
            weights[i], losses[i] = least_squares_SGD(y_train, X_train.T , initial_w, max_iters, gamma)
        if method == 'least_squares':
            weights[i], losses[i] = least_squares(y_train,  X_train.T)
        if method == 'ridge_regression':
            weights[i], losses[i] = ridge_regression(y_train, X_train.T, lambda_)
        if method == 'logistic_regression':
            weights[i], losses[i] = logistic_regression(y_train, X_train.T, initial_w, max_iters, gamma)
        if method == 'reg_logistic_regression':
            weights[i], losses[i] = reg_logistic_regression(y_train, X_train.T, lambda_, initial_w, max_iters, gamma)
        
        # The model is tested with a MSE computation on the last fold
        MSE[i] = compute_loss_MSE(y_test, X_test.T, weights[i])
    
    # To dexcribe the model, we average weights, losses and MSE
    mean_weights = np.mean(weights, axis=0) 
    mean_losses = np.mean(losses, axis=0) 
    mean_MSE = MSE.mean()

    return mean_weights, mean_losses, mean_MSE
    
def normal_init_w(size, mu = 0, sigma = 0.1):
    return np.random.normal(mu, sigma, size)