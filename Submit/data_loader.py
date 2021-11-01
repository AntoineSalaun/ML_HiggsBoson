import numpy as np


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
    
    #----------- Splitting data --------------------------------
    
    if jet_split: 
        data_norm = [data_norm[:,i] for i in jet_mask_array]    
    
    if train :        
        return np.array(data_norm), np.array(target_onehot)
    else :
        return np.array(data_norm)
    
    