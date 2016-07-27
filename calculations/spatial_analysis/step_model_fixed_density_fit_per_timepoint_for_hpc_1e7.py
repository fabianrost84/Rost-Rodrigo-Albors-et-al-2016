# coding: utf-8
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pylab as plt
import scipy as sp
import pandas as pd
import rost
import os
import pymc
import shutil
import multiprocessing
import datetime

# In[8]:
print os.getcwd()
cell_number_data_file = '../../data/cell_number_data.csv'
# cell_number_data_file = '../../140204_create_test_data/140204_step_data/step_data.pkl'

cell_number_data = pd.read_csv(cell_number_data_file)


# In[9]:

cell_number_data['GF'] = cell_number_data['PCNA'] / cell_number_data['SOX2']
cell_number_data['mi'] = cell_number_data['m'] / cell_number_data['PCNA'] * 13.2 / 50.



# ## pymc model

# In[17]:

outgrowth = pd.Series([0.0, 56.5, 163.28571428571428, 451.75, 1278.5, 2257.25],
                      index = pd.Index([0.0, 2.0, 3.0, 4.0, 6.0, 8.0], name = 'time'),
                      name = 'outgrowth')
                         

def make_model(data, mi_mean_min, mi_mean_max, GF_mean_min, GF_mean_max, constant_proliferation = False):
    values_SOX2 = {}
    values_m = {}
    values_nonPCNA = {}
    switchpoint = {}
    mi_left = {}
    GF_left = {}
    SOX2_mean_left = {}
    mi_right = {}
    GF_right = {}
    SOX2_mean_right = {}
    cells_SOX2_float = {}
    cells_nonPCNA = {}
    cells_m = {}

    
    ls = 50.0 # length of section
    l = pd.read_csv('../../data/cell_length_data.csv')['cell_length'].mean()  # length of cell
    
    def step_function(x, switchpoint, left_value, right_value):
        ''' This function should return something in the same format as the passed array 

          Specifically, it produces an output that has an array of the same size of the experimental data
        but whose contents are the lower average until the switchpoint, and the upper average past the switchpoint.
        For all purposes, this builds the model to which we want to compare the data.
        '''
        return sp.where(x<=switchpoint, left_value, right_value)

    def ma(array, fill_value):
        return sp.ma.masked_array(array, sp.isnan(array), fill_value = fill_value)
  


    #data = data.dropna(how='all', subset = ['m', 'PCNA', 'SOX2'])
    
    # I'll drop all nan because of the potential bug with the binomials (see my question on stackoverflow)
    data = data.dropna(how='all', subset = ['m', 'PCNA', 'SOX2'])
    data = data.sort_values(['ID', 'pos'])
    
    # priors for global mean values
    
    # define priors for left side of step function
    mi_left_pop= pymc.Uniform('mi_left_pop', lower = mi_mean_min, upper = mi_mean_max, value = 0.02)
    GF_left_pop = pymc.Uniform('GF_left_pop', lower = GF_mean_min, upper = GF_mean_max, value = 0.8)

    # define priors for right side of step function
    if constant_proliferation:
        mi_right_pop = mi_left_pop
        GF_right_pop = GF_left_pop
    else:
        mi_right_pop = pymc.Uniform('mi_right_pop', lower = mi_mean_min, upper = mi_mean_max, value = 0.04)
        GF_right_pop = pymc.Uniform('GF_right_pop', lower = GF_mean_min, upper = GF_mean_max, value = 0.9)
        # stepsizes
        @pymc.deterministic(name='step_mi', plot=True)
        def step_mi(mi_left = mi_left_pop, mi_right = mi_right_pop):
            return mi_right - mi_left

        @pymc.deterministic(name='step_GF', plot=True)
        def step_GF(GF_left = GF_left_pop, GF_right = GF_right_pop):
            return GF_right - GF_left

    
    # prior distribution for sigma beeing uniformly distributed
    GF_sigma_inter = pymc.Uniform('GF_sigma_inter', lower = 0.001, upper = 0.2)
    mi_sigma_inter = pymc.Uniform('mi_sigma_inter', lower = 0.001, upper = 0.2)

    
    # switchpoint
    if not constant_proliferation:
        switchpoint_pop = pymc.Uniform('switchpoint_pop',
                                       lower = -2000,
                                       upper = outgrowth[data['time'].iloc[0]], 
                                       value = -500)
        switchpoint_sigma_inter = pymc.Uniform('switchpoint_sigma_inter', lower=1.0, upper=400.0, value = 50)
    
    
    for ID, IDdata in data.groupby('ID'):
        values_SOX2[ID] = ma(IDdata['SOX2'], 35.5)
        values_nonPCNA[ID] = ma(IDdata['SOX2'] - IDdata['PCNA'], 3.5)
        values_m[ID] = ma(IDdata['m'], 1.5)
        
        # Model definition

        #priors
        # switchpoint[ID]: for all observables
        
        if constant_proliferation:
            switchpoint[ID] = 0.0
        else:
            switchpoint[ID] = pymc.Normal('switchpoint_{0}'.format(ID), mu = switchpoint_pop,                                          tau = 1/switchpoint_sigma_inter**2, value = -500,
                                         plot = False)
            

        # number of SOX2 cells
        SOX2_mean = sp.mean(values_SOX2[ID])
        SOX2_std = sp.std(values_SOX2[ID])


        # define priors for left side of step function
        mi_left[ID] = pymc.TruncatedNormal('mi_left_{0}'.format(ID), mu = mi_left_pop, tau = 1.0 / mi_sigma_inter**2,
                                           a = 0.0, b = 1.0,
                                  value = 0.02, plot = False)
        GF_left[ID] = pymc.TruncatedNormal('GF_left_{0}'.format(ID), mu = GF_left_pop, tau = 1.0 / GF_sigma_inter**2,
                                           a = 0.0, b = 1.0,
                                  value = 0.5, plot = False)
        

        # define priors for right side of step function
        mi_right[ID] = pymc.TruncatedNormal('mi_right_{0}'.format(ID), mu = mi_right_pop, tau = 1.0 / mi_sigma_inter**2,
                                            a = 0.0, b = 1.0,
                                            value = 0.02, plot = False)
        GF_right[ID] = pymc.TruncatedNormal('GF_right_{0}'.format(ID), mu = GF_right_pop, tau = 1.0 / GF_sigma_inter**2,
                                            a = 0.0, b = 1.0,
                                            value = 0.5, plot = False)
    
        
        # step functions
        @pymc.deterministic(name='mi_{}'.format(ID))
        def mi(positions = sp.array(IDdata['pos']), switchpoint = switchpoint[ID],
               left_value = mi_left[ID], right_value = mi_right[ID]):
            return step_function(positions, switchpoint, left_value, right_value)

        @pymc.deterministic(name='GF_{}'.format(ID))
        def GF(positions = sp.array(IDdata['pos']), switchpoint = switchpoint[ID],
               left_value = GF_left[ID], right_value = GF_right[ID]):
            return step_function(positions, switchpoint, left_value, right_value)

        @pymc.deterministic(name='SOX2_mean_{}'.format(ID))
        def SOX2_mean(positions = sp.array(IDdata['pos']), switchpoint = switchpoint[ID],
                      left_value = SOX2_mean , right_value = SOX2_mean):
            return step_function(positions, switchpoint, left_value, right_value)

        #likelihoods
        cells_SOX2_float[ID] = pymc.Normal('cells_SOX2_float_{0}'.format(ID), mu=SOX2_mean, tau = 1/SOX2_std**2, value = values_SOX2[ID],                                           plot = False, observed = True)


        @pymc.deterministic(name='cells_SOX2_{}'.format(ID))
        def cells_SOX2(csf = cells_SOX2_float[ID]):
            return sp.around(csf)




        cells_nonPCNA[ID] = pymc.Binomial('cells_nonPCNA_{0}'.format(ID),                                        n = cells_SOX2,                                        p = (1.0 - GF),                                        value = values_nonPCNA[ID], observed = True, plot = False )

        @pymc.deterministic(name='cells_PCNA_{}'.format(ID))
        def cells_PCNA(cnp = cells_nonPCNA[ID], cs = cells_SOX2):
            return  cs - cnp



        @pymc.deterministic(name='cells_PCNA_section_{}'.format(ID))
        def cells_PCNA_section(cp = cells_PCNA, ls = ls, l = l):
            return cp * ls / l



        cells_m[ID] = pymc.Binomial('cells_m_{0}'.format(ID),                                n = cells_PCNA_section,                                p = mi,                                value = values_m[ID], observed = True, plot = False)



    
    values_SOX2 = pymc.Container(values_SOX2)
    values_SOX2 = pymc.Container(values_SOX2)
    values_m = pymc.Container(values_m)
    values_nonPCNA = pymc.Container(values_nonPCNA)
    switchpoint = pymc.Container(switchpoint)
    mi_left = pymc.Container(mi_left)
    GF_left = pymc.Container(GF_left)
    SOX2_mean_left = pymc.Container(SOX2_mean_left)
    mi_right = pymc.Container(mi_right)
    GF_right = pymc.Container(GF_right)
    SOX2_mean_right = pymc.Container(SOX2_mean_right)
    cells_SOX2_float = pymc.Container(cells_SOX2_float)
    cells_nonPCNA = pymc.Container(cells_nonPCNA)
    cells_m = pymc.Container(cells_m)

    return locals()



# ## Fit the real data

# In[29]:

GF_mean_min = (cell_number_data['PCNA'] / cell_number_data['SOX2']).min()
GF_mean_max = (cell_number_data['PCNA'] / cell_number_data['SOX2']).max()

mi_mean_min = 0.0
mi_mean_max = 0.1


burn = 1e6
iter_ = 1e7+1e6
thin = 100

dir_ = '{0}_{1}_{2}'.format(cell_number_data_file.split('.')[-2].split('/')[-1], datetime.datetime.now().strftime("%y%m%dT%H%M%S"), int(iter_-burn) )
out_path = os.path.join('results', dir_)

try:
    shutil.rmtree(out_path)
    print('Removed previous results')
except:
    print('No previous results to remove?!')
rost.mkdir_p(out_path)

meta = pd.Series()
meta['datafile'] = cell_number_data_file
meta['burn'] = burn
meta['iter_'] = iter_
meta['thin'] = thin
meta.to_csv(os.path.join(out_path, 'meta.txt'), sep = '\t')

cell_number_data.to_pickle(os.path.join(out_path, 'cell_number_data.pkl'))

def fit_model((ID, data)):
    print ID    

    M = pymc.MCMC(make_model(data, mi_mean_min, mi_mean_max, GF_mean_min, GF_mean_max), db='hdf5', dbname = os.path.join(out_path, '{0}.hdf5'.format(ID)))
    M.sample(iter=iter_, burn=burn, thin=thin, progress_bar=False)
    print()
    pymc.Matplot.plot(M, path = out_path, suffix = '_{0}'.format(ID));
    plt.close('all')
    M.db.close()


l = [[ID, data] for ID, data in cell_number_data.groupby('time')]

p = multiprocessing.Pool(processes=5)

p.map(fit_model, l)

