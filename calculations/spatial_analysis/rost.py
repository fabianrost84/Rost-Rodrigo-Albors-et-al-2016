import numpy as np
import sys
import errno
import os
import scipy as sp
from scipy import stats
import pandas as pd
import IPython
from IPython import display
import matplotlib.pyplot as plt


# OS & files

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST:
            pass
        else: raise
      
# ipython publication

def print_data_for_import(data, name):
    """ Print code that generates a pandas dataframe with the given name and with the content of the given pandas dataframe
    """
    print('{0} = pd.DataFrame()'.format(name))
    for column in data:
        print('{0}[\'{1}\'] = {2}'.format(name, column, list(data[column])).replace('nan', 'sp.nan'))        

# statistics

def nanmean(array):
  """ Gives back scipy's nanmean for arrays, gives nan for empty arrays
  """
  array = np.array(array)
  if len(array)>0:
      mean = spnanmean(array)
  else:
      mean = np.nan
  return mean

def nanmean_empty(a):
  """ a must be an array!
      Gives back scipy's nanmean for arrays, gives nan for empyty arrays.
      An array is defined as empty by list(array)==[].
  """
  if a.size==0:
    mean = np.nan
  else:
    mean = spnanmean(array)
  return mean
  
def nanstd_empty(a):
  """ a must be an array!
      Gives back scipy's nanstd for arrays, gives nan for empyty arrays.
      An array is defined as empty by list(array)==[].
  """
  if a.size==0:
    std = np.nan
  else:
    std = spnanstd(array)
  return std

def correlation(x, y, xlabel='x', ylabel='y'):
    """ display some information about x and y.
        x and y should be 1D-arrays of the same length.
        Output: a scatter plot of x and y, Pearson and Spearman correlation coefficients and p-values
    
    """
  
    plt.title(xlabel + ', ' + ylabel, fontsize = 24, fontweight = 'bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y);
    
    plt.show()
    
    # statistical tests
    pearson_coef,  pearson_p  = sp.stats.pearsonr(x, y)
    spearman_coef, spearman_p = sp.stats.spearmanr(x, y)
    
    correl_test_results = pd.DataFrame()
    correl_test_results.loc['Pearson', 'Coefficient'] = pearson_coef
    correl_test_results.loc['Pearson', 'p-value'] = pearson_p
    correl_test_results.loc['Spearman', 'Coefficient'] = spearman_coef
    correl_test_results.loc['Spearman', 'p-value'] = spearman_p

    IPython.display.display(correl_test_results)
    
# Baysian inference

def conf_interval(trace, cl = 0.95, borders = (None, None) ):
  """
      Find the narrowst confidence interval for a given trace of a paramter at given confidence level cl
      Returns the lower and upper boundary of the confidence interval
      
      It's possible to pass a tuple with the borders (bmin, bmax) of the prior interval. Then the confidence interval limits will be set to bmin or bmax if the calculated confidence interval is one-sided (this means e.g. that the calculated minimum border of the confidence interval equals the minimal value of the trace)
  """
  trace = trace.copy()
  trace.sort()
  N = int(cl * len(trace))
  
  interval_lengths = np.array([trace[i:N+i][-1] - trace[i:N+i][0] for i in range(len(trace) - N)])
  
  i_min = interval_lengths.argmin()
  
  p_min = trace[i_min]
  p_max = trace[i_min + N]
  
  if type(borders[0]) == float:
    if p_min == trace.min():
      p_min = borders[0]
  if type(borders[1]) == float:
    if p_max == trace.max():
      p_max = borders[1]
    
  
  return p_min, p_max
  