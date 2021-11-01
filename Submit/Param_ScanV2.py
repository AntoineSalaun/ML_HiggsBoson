from Utility import *
from helpers import *
from implementation import *
import numpy as np

def parameter_scan(y,tx,w,max_iters,method,lambda_fixed,gamma_fixed,lambda_array,gamma_array):
    #Returns arrays of losses for different lambdas and gammas, for different methods of regression and classification
    #y : output data.
    #tx : input data.
    #w : initial weights needed for some regression method.
    #max_iters : maximum number of iterations a method can do if it has not converged before.
    #method : string that enables one to select on which method to perform the Parameter Scan.
    #lambda_fixed & gamma_fixed : fixed scalar needed to run some methods, while scaning the other parameter.
    #lambda_array & gamma_array : arrays of lambda & gammas respectively to scan on the method in order to choose the best penalty term & the optimum step size.
    
    """What will be returned by the function """
    losses_lambda=[]
    losses_gamma=[]
    """Intermidiate variables needed for filling the two arrays above"""
    weight_lambda=None
    loss_lambda=None
    weight_gamma=None
    loss_gamma=None
    #Beginning of the Parameter Scan
    if (method!='least_squares'):
        if (method=='ridge_regression'):
            for l in lambda_array:
                weight_lambda,loss_lambda=ridge_regression(y,tx,l)
                #print(loss_lambda)
                losses_lambda.append(loss_lambda)
            weight_gamma,loss_gamma=ridge_regression(y,tx,lambda_fixed)
            #print(loss_gamma)
            np.append(losses_gamma,loss_gamma)
            string='ridge_regression'
        if (method=='regularized_logistic_regression'):
            for l in lambda_array:
                weight_lambda,loss_lambda=reg_logistic_regression(y,tx,l,w,max_iters,gamma_fixed)
                #print(loss_lambda)
                losses_lambda.append(loss_lambda)
            for g in gamma_array:
                weight_gamma,loss_gamma=reg_logistic_regression(y,tx,lambda_fixed,w,max_iters,g)
                #print(loss_gamma)
                losses_gamma.append(loss_gamma)
            #print("regularized_logistic_regression\n")
            string='regularized_logistic_regression'
        if (method=='logistic_regression'):
            for g in gamma_array:
                weight_gamma,loss_gamma=logistic_regression(y,tx,w,max_iters,g)
                #print(loss_gamma)
                losses_gamma.append(loss_gamma)
            #print("logistic_regression\n")
            string='logistic_regression'
        if (method=='least_squares_GD'):
            for g in gamma_array:
                weight_gamma,loss_gamma=least_squares_GD(y,tx,w,max_iters,g)
                #print(loss_gamma)
                losses_gamma.append(loss_gamma)
            #print("least_squares_GD\n")
            string='least_squares_GD'
        if (method=='least_squares_SGD'):
            for g in gamma_array:
                weight_gamma,loss_gamma=least_squares_SGD(y,tx,w,max_iters,g)
                #print(loss_gamma)
                losses_gamma.append(loss_gamma)
            #print("least_squares_SGD\n")
            string='least_squares_SGD'
    else:
        weight_lambda,loss_lambda=least_squares(y,tx)
        #print(loss_lambda)
        losses_lambda.append(loss_lambda)
        losses_gamma.append(loss_lambda)
        #print("least_squares\n")
        string='least_squares'
            
    return losses_lambda, losses_gamma, string