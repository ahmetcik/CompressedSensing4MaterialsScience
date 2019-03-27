import pandas as pd
import numpy as np
from itertools import combinations
from time import time
import matplotlib.pyplot as plt 
from sklearn.linear_model import Lasso
import scipy.stats as ss
import warnings
from collections import Counter
from bokeh.io import output_notebook

import ase.io
import nglview as nv

from modules.sisso import SissoRegressor
from modules.combine_features import combine_features
from modules.viewer import Viewer

# load data
df = pd.read_pickle("data/data.pkl")

# print data without structure objects
df.drop(['struc_obj_RS', 'struc_obj_ZB', 'struc_obj_min'], axis=1)

example_structure = df.loc['AgBr', 'struc_obj_RS']
view = nv.show_ase(example_structure *[3, 3, 3] )

def get_data(selected_feature_list, allowed_operations):
    # add both '(A)', '(B)' to each feature
    selected_featureAB_list = [f+A_or_B for f in selected_feature_list for A_or_B in ['(A)', '(B)']]
    
    # extract energy differences and selected features from df_data 
    P = df['energy_diff'].values
    df_features = df[selected_featureAB_list]
    
    
    # derive new features using allowed_operations
    df_combined = combine_features(df=df_features, allowed_operations=allowed_operations, is_print=False)
    return P, df_combined

# selected_feature_list = ['IP', 'EA', 'E_HOMO', 'E_LUMO', 'r_s', 'r_p', 'r_d', 'Z', 'period', 'd']
selected_feature_list = ['r_s', 'r_p']

# allowed_operations = ['+', '-', '|-|', '*', '/' '^2', '^3',  'exp']
allowed_operations = ['+']

P, df_D = get_data(selected_feature_list, allowed_operations)

# print derived features
df_D
def L0(P, D, dimension):
    n_rows, n_columns = D.shape
    D = np.column_stack((D,np.ones(n_rows)))
    SE_min = np.inner(P,P)
    coef_min, permu_min = None, None
    for permu in combinations(range(n_columns),dimension):
        D_ls = D[:,permu+(-1,)]
        coef, SE, __1, __2 = np.linalg.lstsq(D_ls,P)
        try:
            if SE[0] < SE_min: 
                SE_min = SE[0]
                coef_min, permu_min = coef, permu
        except:
            pass
    RMSE = np.sqrt(SE_min/n_rows)
    return RMSE, coef_min, permu_min

selected_feature_list = ['r_s', 'r_p', 'r_d', 'EA', 'IP']
allowed_operations = []

P, df_D = get_data(selected_feature_list, allowed_operations)
features_list = df_D.columns.tolist()
D = df_D.values

for dim in range(1,11):
    RMSE, coefficients, selected_indices = L0(P,D,dim)
    dim, RMSE, [features_list[i] for i in selected_indices]

f, (ax1, ax2) = plt.subplots(1,2, sharex=True, figsize=(12,8))
ax1.set_xlabel('Feature space size')
ax2.set_xlabel('Feature space size')
ax1.set_ylabel('RMSE [eV]')
ax2.set_ylabel('Time [s]')
#ax2.set_yscale('log')


def lasso_fit(lam, P, D, feature_list):
    #LASSO
    D_standardized = ss.zscore(D)
    lasso =  Lasso(alpha=lam)
    lasso.fit(D_standardized, P)
    coef =  lasso.coef_
    
    # get strings of selected features
    selected_indices = coef.nonzero()[0]
    selected_features = [feature_list[i] for i in selected_indices]
    
    # get RMSE of LASSO model
    P_predict = lasso.predict(D_standardized)
    RMSE_LASSO = np.linalg.norm(P-P_predict) / np.sqrt(82.)

    #get RMSE for least-square fit
    D_new = D[:, selected_indices]
    D_new = np.column_stack((D_new, np.ones(82)))
    RMSE_LS = np.sqrt(np.linalg.lstsq(D_new,P)[1][0]/82.)
        
    return RMSE_LASSO, RMSE_LS, coef, selected_features

#import Data
selected_feature_list = ['r_s', 'r_p', 'r_d', 'EA', 'IP']
allowed_operations = ['+','|-|','exp', '^2']
P, df_D = get_data(selected_feature_list, allowed_operations)
D = df_D.values
features_list = df_D.columns.tolist()

# change lam between 0.02 and 0.34, e.g. 0.34, 0.30, 0.20, 0.13, 0.10, 0.02
lam = 0.2

RMSE_LASSO, RMSE_LS, coef, selected_features = lasso_fit(lam, P, D, features_list)
plt.bar(range(len(coef)), np.abs(coef))
plt.xlabel("Coefficient index $i$")
plt.ylabel("$|c_i|$")
sisso = SissoRegressor(n_nonzero_coefs=3, n_features_per_sis_iter=10)

sisso.fit(D, P)

# parameters for feature space construction
selected_feature_list = ['IP', 'EA', 'r_s', 'r_p','r_d']
allowed_operations = ['+','|-|','exp','^2', '/']

# get the data
P, df_D = get_data(selected_feature_list, allowed_operations)
D = df_D.values
features_list = df_D.columns.tolist()

sisso = SissoRegressor(n_nonzero_coefs=3, n_features_per_sis_iter=10)

sisso.fit(D, P)

view = Viewer(show_geos=True)
view.show_map(df, df_D, sisso.l0_selected_indices[1], is_show=False)

def split_data(P, D, cv_i):
    P_1, P_test, P_2 = np.split(P, [cv_i, cv_i+1])
    P_train = np.concatenate((P_1,P_2))
    D_1, D_test, D_2 = np.split(D, [cv_i, cv_i+1])
    D_train = np.concatenate((D_1,D_2))
    return P_train, P_test, D_train, D_test

n_compounds = len(P)
dimensions = range(1,4)
features_count = [[] for i in range(3)]
P_predict = np.empty([len(dimensions), n_compounds])

sisso = SissoRegressor(n_nonzero_coefs=3, n_features_per_sis_iter=5)

for cv_i in range(1):
    P_train, P_test, D_train, D_test = split_data(P, D, cv_i)
        
    sisso.fit(D_train, P_train)
    for dim in dimensions:      
        features = [features_list[i] for i in sisso.l0_selected_indices[dim - 1]]
        predicted_values = sisso.predict(D_test, dim=dim)
        
        features_count[dim-1].append( tuple(features) )        
        P_predict[dim-1,cv_i] = predicted_values

print "Test run successful"
