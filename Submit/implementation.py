from helpers import *
import numpy as np
from Utility import *

    
def least_squares(y, tx):
    """
    Computes a least squares linear regression on the data
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    
    Returns
    ----------
    w : Least Squares optimized weights [d]
    loss : Loss for the optimal weights (scalar)
    """
    w = np.linalg.solve(tx.T@tx,tx.T@y)
    loss = compute_loss_MSE(y, tx, w)
    return w,loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Computes a linear regression using gradient descent to output the optimal weights 
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    max_iters : maximum iterations (scalar)
    gamma : step size (scalar)
    
    Returns
    ----------
    w : Optimal weights [d]
    loss : Loss for the optimal weights (scalar)
    """
    w_list = [initial_w]
    loss_list = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss_MSE(y,tx,w)
        w = w - gamma * gradient
        w_list.append(w)
        loss_list.append(loss)
    return w_list[-1], loss_list[-1]
    

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Computes a linear regression using stochastic gradient descent to output the optimal weights 
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    max_iters : maximum iterations (scalar)
    gamma : step size (scalar)
    
    Returns
    ----------
    w : Optimal weights [d]
    loss : Loss for the optimal weights (scalar)
    """
    w_list = [initial_w]
    loss_list = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_n,tx_n in batch_iter(y, tx, 1):
            gradient = compute_stoch_gradient(y_n, tx_n, w)
            w = w - gamma * gradient
            loss = compute_loss_MSE(y_n, tx_n, w)
            w_list.append(w)
            loss_list.append(loss)
    return w_list[-1], loss_list[-1]
       



def ridge_regression(y, tx, l):
    """
    Computes a linear regression using gradient descent with an L2 penalty top optimize the weights 
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    l : regularization strength

    
    Returns
    ----------
    w : Least Squares optimized weights [d]
    loss : Loss for the optimal weights (scalar)
    """
    l_ = 2*l*tx.shape[0]         
    I = np.eye(tx.shape[1])             

    w = np.linalg.solve(tx.T@tx + l_*I, tx.T@y)
    loss = compute_loss_MSE(y, tx, w)
    return w, loss



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Computes a logistic regression using gradient descent to output the optimal weights 
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    max_iters : maximum iterations (scalar)
    gamma : step size (scalar)
    
    Returns
    ----------
    w : Optimal weights [d]
    loss : Loss for the optimal weights (scalar)
    """
    y_resize = (1+y)/2                        #rescales target so that -1 values are changed to 0 
    w_list = [initial_w]
    loss_list = []
    w = initial_w
    
    for n_iter in range(max_iters):
        grad = calculate_gradient_LR(y_resize, tx, w)
        w = w - gamma * grad
        loss =  compute_loss_LG(y_resize, tx, w)
        w_list.append(w)
        loss_list.append(loss)
    return w_list[-1],loss_list[-1]


def reg_logistic_regression(y, tx, l, initial_w, max_iters, gamma):
    """
    Computes a regularized logistic regression using gradient descent to output the optimal weights 
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    l : regularization strength
    max_iters : maximum iterations (scalar)
    gamma : step size (scalar)
    
    Returns
    ----------
    w : Optimal weights [d]
    loss : Loss for the optimal weights (scalar)
    """
    y_resize = (1+y)/2                        #rescales target so that -1 values are changed to 0 
    w_list = [initial_w]
    loss_list = []
    w = initial_w

    for n_iter in range(max_iters):
        grad = calculate_gradient_LR(y_resize, tx, w) + 2*l*w
        w = w - gamma*grad
        loss =  compute_loss_LG(y_resize, tx, w)+ l*np.linalg.norm(w)
        w_list.append(w)
        loss_list.append(loss)
        if (n_iter > 1) and (np.abs(loss_list[-1] - loss_list[-2]) <= 1e-8):
            break
    return w_list[-1],loss_list[-1]
    
