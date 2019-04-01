import os
import pandas as pd
import numpy as np
from itertools import combinations
from time import time
import matplotlib.pyplot as plt 
from sklearn.linear_model import Lasso
import scipy.stats as ss
import warnings
from collections import Counter
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, LeaveOneOut

from modules.sisso import SissoRegressor
from modules.combine_features import combine_features
from modules.viewer import show_structure, show_map
warnings.filterwarnings('ignore')

# load data
df = pd.read_pickle("data/data.pkl")

# print data without structure objects
df.drop(['struc_obj_RS', 'struc_obj_ZB', 'struc_obj_min'], axis=1)

example_structure = df.loc['AgBr', 'struc_obj_RS'] * [3, 3, 3]
#show_structure(example_structure)

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
        coef, SE, __1, __2 = np.linalg.lstsq(D_ls,P,rcond=-1)
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
    RMSE_LS = np.sqrt(np.linalg.lstsq(D_new,P, rcond=-1)[1][0]/82.)
        
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

# get 2d solution
P_predict = sisso.predict(D, dim=2)
D_selcted = df_D.values[:, sisso.l0_selected_indices[1]]
features = df_D.columns[sisso.l0_selected_indices[1]]

# plot 2D map 
#show_map(df, D_selcted, P_predict, features)

# Leave-one-out cross-validation
chemical_formulas = df_D.index.tolist()

n_compounds = len(P)
dimensions = range(1, 4)
features_count = [[] for i in range(3)]
P_predict = np.empty([len(dimensions), n_compounds])

sisso = SissoRegressor(n_nonzero_coefs=3, n_features_per_sis_iter=1)
loo = LeaveOneOut()

for indices_train, index_test in loo.split(P):
    i_cv = index_test[0]
        
    sisso.fit(D[indices_train], P[indices_train])
    
    for dim in dimensions:      
        features = [features_list[i] for i in sisso.l0_selected_indices[dim - 1]]
        predicted_value = sisso.predict(D[index_test], dim=dim)[0]
        
        features_count[dim-1].append( tuple(features) )        
        P_predict[dim-1, i_cv] = predicted_value
        
prediction_errors = np.linalg.norm(P-P_predict, axis=1)/np.sqrt(n_compounds)

#kernel ridge
selected_feature_list = ['IP', 'EA', 'r_s', 'r_p','r_d']
allowed_operations = []

P, df_D = get_data(selected_feature_list, allowed_operations)
features_list = df_D.columns.tolist()
D = df_D.values


kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=2,
                  param_grid={"alpha": np.logspace(-3, 0, 2),
                              "gamma": np.logspace(-2, 1, 2)})
P_predict_kr = []
loo = LeaveOneOut()
for train_indices, test_index in loo.split(P):
    kr.fit(D[train_indices], P[train_indices])
    P_predict_kr.append(kr.predict(D[test_index])[0])

prediction_rmse_kr = np.linalg.norm(np.array(P_predict_kr) - P)/np.sqrt(P.size)

maxi = max(max(P), max(P_predict_kr))
mini = min(min(P), min(P_predict_kr))
plt.plot([maxi,mini], [maxi,mini], 'k')
plt.scatter(P, P_predict[-1], label='SISSO 3D, RMSE = %.3f eV/atom' % prediction_errors[dim-1])
plt.scatter(P, P_predict_kr,  label='KR, RMSE = %.3f eV/atom' % prediction_rmse_kr)
plt.xlabel('E_diff_DFT'), plt.ylabel('E_diff_predicted')
plt.legend()


print("Test run successful")
