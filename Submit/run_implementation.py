from Utility import *
from implementation import *
import numpy as np



def main():
    #Definition of parameters
    
    lambda_r = 1e-4
    gamma = 1e-3
    lambda_log = 0.5#1e-5
    
    N_sample = 15000
    N_test = 2000
    
    # If the data are in the 'data' folder in the same folder as this code, the importation of the data or else specify FILEPATH as 1st argument
    data,target = data_loader(jet_split = False) 
    data_s = data[0:N_sample,:]
    target_s = target[0:N_sample]
    initial_w = np.random.randn(data_s.shape[1])
    
    X_test = data[N_sample:N_sample + N_test,:]
    y_test = target[N_sample:N_sample + N_test]
    
    
    w_LS, l_LS = least_squares_GD(target_s, data_s, initial_w, 1000, gamma )
    w_LSn, l_LSn = least_squares(target_s, data_s)
    w_SGD, l_SGD = least_squares_SGD(target_s, data_s, initial_w, 1000, gamma)
    w_r, l_r = ridge_regression(target_s, data_s, lambda_r)
    w_log, l_log = logistic_regression(target_s, data_s, initial_w, 1000, gamma)
    w_log2, l_log2 = reg_logistic_regression(target_s, data_s, lambda_log, initial_w, 1000, gamma)
    
    
    
    print('Accuracy for least squares with gradient descent : ', accuracy(y_test, X_test, w_LS), '% \n')
    print('Accuracy for least squares : ', accuracy(y_test, X_test, w_LSn), '% \n')
    print('Accuracy for least squares with stochastic gradient descent : ', accuracy(y_test, X_test, w_SGD), '% \n')
    print('Accuracy for least squares with Ridge regression : ', accuracy(y_test, X_test, w_r), '% \n')
    print('Accuracy for least squares with logistic regression : ', accuracy(y_test, X_test, w_log), '% \n')
    print('Accuracy for least squares with regularized logistic regression : ', accuracy(y_test, X_test, w_log2), '% \n')

    print('(It is to note that here it is only a test of the implementations of the functions, it is not the optimal results as we did not do a jet split here)')

if __name__ == "__main__": main()