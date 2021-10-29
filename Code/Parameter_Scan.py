from Utility import *
from helpers import *
import numpy as np

def parameter_scan(y,tx,w,max_iters,lambda_fixed,gamma_fixed,lambda_array,gamma_array):
    
    losses_ridge_lambda=[]
    losses_logistic_reg_lambda=[]
    
    losses_ridge_gamma=[]
    losses_logistic_reg_gamma=[]
    losses_logistic_gamma=[]
    losses_GD_gamma=[]
    losses_SGD_gamma=[]
    
    for l in range(lambda_array):
        weight_ridge_lambda,loss_ridge_lambda=ridge_regression(y,tx,l)
        losses_ridge_lambda.append(loss_ridge_lambda)
        
        weight_logistic_reg_lambda,loss_logistic_reg_lambda=reg_logistic_regression(y,tx,l,w,max_iters,gamma_fixed)
        losses_logistic_reg_lambda.append(loss_logistic_reg_lambda)
    
    weight_ridge_gamma,loss_ridge_gamma=ridge_regression(y,tx,lambda_fixed)
    losses_ridge_gamma.append(loss_ridge_gamma)
    
    for g in range(gamma_array):
        
        weight_logistic_reg_gamma,loss_logistic_reg_gamma=reg_logistic_regression(y,tx,lambda_fixed,w,max_iters,g)
        losses_logistic_reg_gamma.append(loss_logistic_reg_gamma)
        
        weight_logistic_gamma,loss_logistic_gamma=logistic_regression(y,tx,w,max_iters,g)
        losses_logistic_gamma.append(loss_logistic_gamma)
        
        weight_GD_gamma,loss_GD_gamma=least_squares_GD(y,tx,w,max_iters,g)
        losses_GD_gamma.append(loss_GD_gamma)
        
        weight_SGD_gamma,loss_SGD_gamma=least_squares_SGD(y,tx,w,max_iters,g)
        losses_SGD_gamma.append(loss_SGD_gamma)
    return losses_ridge_lambda,losses_logistic_reg_lambda,losses_ridge_gamma,losses_logistic_reg_gamma,losses_logistic_gamma,losses_GD_gamma,losses_SGD_gamma