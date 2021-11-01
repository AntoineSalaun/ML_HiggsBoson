import numpy as np


#-------------------Loss Functions--------------------------

def compute_loss_MSE(y, tx, w):
    """
    Computes the mean squared error between the target and the estimated values.
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    
    Returns
    Mean squared error : scalar     
    """
    return np.mean((y-tx@w)**2)

def compute_loss_MAE(y, tx, w):
    """
    Computes the mean absolute error between the target and the estimated values.
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    
    Returns
    Mean absolute error : scalar     
    """
    
    N = y.shape[0]
    return np.sum(abs(y-tx@w))/N

def compute_loss_LG(y, tx, w):
    """
    Computes the negative log likelihood between the target and the estimated values.
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    
    Returns
    ----------
    Negative log likelihood : scalar     
    """
    
    N = y.shape[0]
    estim = tx@w
    return np.sum(np.logaddexp(0,estim)-y*estim)/N

#-------------------Logistic regression utility----------------------------

def sigmoid(x):
    """
    Computes the sigmoid function of the input x
    
    Parameters
    ----------
    x : input [m]
    
    Returns
    ----------
    output y = sigmoid(x) [m]
    """
    return np.exp(-np.logaddexp(0,-x))

#-----------------Gradient computation--------------------------

def compute_gradient(y, tx, w):
    """
    Computes the gradient
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    
    Returns
    ----------
    Gradient wrt each weight : array [d]
    """
    N = y.shape[0]
    err = y - tx@w
    return (-1/N) *((tx.T)@err)

def compute_stoch_gradient(y, tx, w):
    """
    Computes a stochasic gradient from a subset of the data
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    
    Returns
    ----------
    Gradient wrt each weight : array [d]     
    """
    N = float(y.shape[0])
    N = y.shape[0]
    err = y - tx@w
    return -1./N * (tx.T@err)

def calculate_gradient_LR(y, tx, w):
    """
    Computes the gradient of the logistic regression's loss 
    
    Parameters
    ----------
    y : target array [n]
    tx : data [n,d]
    w : weights [d]
    
    Returns
    ----------
    Gradient wrt each weight : array [d]     
    """
    return tx.T@( sigmoid(tx@w) - y)






#----------------------------------Loading the data-------------------------------------------

def data_loader(FILEPATH = 'data', train = True, onehot = True, nan = 'Mean', jet_split = False):
    '''
    Function for data loading and pre processing of the data. 
    Many parameters can be given in order to modulate the preprocessing so 
    that the optimal preprocessing can be found for the learing of the parameters.
    
    Parameters
    ----------
    FILEPATH : Relative path to the train.csv & test.csv files
    train : Boolean 
    onehot : Boolean for one hot encoding or with {-1;1} for the targets 
    nan : String to check if the data should be recentered w.r.t the mean or the median or just centering with mean with putting 0 instead of NaN.
          (Has to take the values 'Mean', 'Median' or 'Zero')   
    jet_split : Boolean to split data in function of the jet (column 22 of the data) 
    
    Returns
    -------
    if 'train'
    out : ndarray, ndarray
        that represents the data matrix preprocessed and the target one hot encoded
        
    if 'test'
    out: ndarray
        that represents the data matrix preprocessed for the test
    '''
    
    #------------ Data Loading -----------------------------
    if train : 
        data = np.loadtxt(FILEPATH  + "/train.csv",delimiter = ',',skiprows=1, usecols=range(2,32)).T
        target = np.loadtxt(FILEPATH + "/train.csv",delimiter = ',',skiprows=1, dtype = np.dtype('U1'), usecols=1)
    else:
        data = np.loadtxt(FILEPATH  + "/test.csv",delimiter = ',',skiprows=1, usecols=range(2,32)).T
    
   #------------- One Hot encoding for Target ---------------
    if train:
        if onehot :
            s = 0
        else :
            s = -1
        d = {'s':s, 'b':1}
        target_onehot = [d[i] for i in target]
    
    #----------- Mask loading before normalization ----------
    if jet_split:
        jet0_mask = data[22,:] == 0
        jet1_mask = data[22,:] == 1
        jet2_mask = data[22,:] == 2
        jet3_mask = data[22,:] == 3
        jet_mask_array = [jet0_mask, jet1_mask, jet2_mask, jet3_mask]
    #----------- Data pre-processing -------------------------
    data[data == -999] = np.nan
    data_norm = data.copy()
    if (nan == 'Mean') or (nan == 'Zero'):
        data_m = np.nanmean(data,axis = 1)
    elif (nan == 'Median' ):
        data_m = np.nanmedian(data,axis = 1)
    
    data_std = np.nanstd(data, axis = 1)
    
    for i in range(len(data)):
        if (nan == 'Zero'):
            np.nan_to_num(data[i,:], copy = False, nan = 0)
        else:
            np.nan_to_num(data[i,:], copy=False, nan = data_m[i])            
        data_norm[i,:] = (data[i,:] - data_m[i])/data_std[i]
    
    data_norm = data_norm.T
    #----------- Splitting data --------------------------------
    
    if jet_split: 
        data_norm = [data_norm[i,:] for i in jet_mask_array] 
        target_onehot_0 = [i for (i, j) in zip(target_onehot, jet0_mask) if j]
        target_onehot_1 = [i for (i, j) in zip(target_onehot, jet1_mask) if j]
        target_onehot_2 = [i for (i, j) in zip(target_onehot, jet2_mask) if j]
        target_onehot_3 = [i for (i, j) in zip(target_onehot, jet3_mask) if j]
        target_onehot = [target_onehot_0, target_onehot_1, target_onehot_2, target_onehot_3]
    if train :        
        return np.array(data_norm), np.array(target_onehot)
    else :
        return np.array(data_norm)
    
    
    
#------------------------Plot the results, esp. the Parameter Scan--------------------------------------------------


def lambda_Param_Scan_visualization(lambds,method, mse_tr0):
    """visualization the curves of mse_tr with respect to lambda ."""
    plt.semilogx(lambds, mse_tr0, marker=".", color='b', label='train error')
    plt.xlabel("lambda")
    plt.ylabel("mse")
    plt.title("Penalty term parameter_scan for "+method)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(method+"lambda")
    
def gamma_Param_Scan_visualization(gamma,method, mse_tr0):
    """visualization the curves of mse_tr with respect to gamma ."""
    plt.semilogx(gamma, mse_tr0, marker=".", color='b', label='train error')
    plt.xlabel("gamma")
    plt.ylabel("mse")
    plt.title("Step-size parameter_scan for "+method)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(method+"gamma")

 #--------------------------Accuracy computation---------------------------------------
def accuracy(y_test, tx_test, weights):
    prediction = tx_test@weights
    prediction_bo = prediction
    for i in range(len(prediction)):
        if prediction[i]<0.5:
            prediction_bo[i] = 0
        if prediction[i]>=0.5:
            prediction_bo[i] = 1
    return(1 - abs(y_test - prediction_bo).mean())
