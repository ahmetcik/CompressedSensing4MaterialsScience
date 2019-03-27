import itertools
import pandas as pd
from math import exp, sqrt
import math

def _my_power_2(row):
    return pow(row[0], 2)         

def _my_power_3(row):
    return pow(row[0], 3)    

def _my_power_m1(row):
    return pow(row[0],-1)

def _my_power_m2(row):
    return pow(row[0],-2)

def _my_power_m3(row):
    return pow(row[0],-3)

def _my_abs_sqrt(row):
    return math.sqrtabs(abs(row[0]))
    
def _my_exp(row):
    return exp(row[0])

def _my_exp_power_2(row):
    return exp(pow(row[0], 2))

def _my_exp_power_3(row):
    return exp(pow(row[0], 3))

def _my_sum(row):
    return row[0] + row[1]
    
def _my_abs_sum(row):
    return abs(row[0] + row[1])

def _my_abs_diff(row):
    return abs(row[0] - row[1])   

def _my_diff(row):
    return row[0] - row[1] 

def _my_div(row):
    return row[0]/row[1]
    
def _my_sum_power_2(row):
    return pow((row[0] + row[1]), 2)

def _my_sum_power_3(row):
    return pow((row[0] + row[1]), 3)
    
def _my_sum_exp(row):
    return exp(row[0] + row[1])

def _my_sum_exp_power_2(row):
    return exp(pow(row[0] + row[1], 2))

def _my_sum_exp_power_3(row):
    return exp(pow(row[0] + row[1], 3))
  
def combine_features(df=None, allowed_operations=None, is_print=True):
    """Generate combination of features given a dataframe and a list of allowed operations.
    
    For the exponentials, we introduce a characteristic energy/length
    converting the 
    ..todo:: Fix under/overflow errors, and introduce handling of exceptions.

    """
    if is_print:
        if allowed_operations:
            print('Selected operations:\n {0}'.format(allowed_operations)) 
        else:
            print('No allowed operations selected.') 
        
    columns_ = df.columns.tolist()    
    
    dict_features = {
        'period':'a0', 
        'Z': 'a0', 
        'group': 'a0', 

        'IP': 'a1', 
        'EA': 'a1', 

        'E_HOMO': 'a2', 
        'E_LUMO': 'a2', 


        'r_s': 'a3',
        'r_p': 'a3',
        'r_d': 'a3',
        'd': 'a3', 
   
        }
        

    df_a0 = df[[col for col in columns_ if dict_features.get(col.split('(', 1)[0])=='a0']].astype('float32')    
    df_a1 = df[[col for col in columns_ if dict_features.get(col.split('(', 1)[0])=='a1']].astype('float32')    
    df_a2 = df[[col for col in columns_ if dict_features.get(col.split('(', 1)[0])=='a2']].astype('float32')    
    df_a3 = df[[col for col in columns_ if dict_features.get(col.split('(', 1)[0])=='a3']].astype('float32')   

    
    col_a0 = df_a0.columns.tolist()
    col_a1 = df_a1.columns.tolist()
    col_a2 = df_a2.columns.tolist()
    col_a3 = df_a3.columns.tolist()

    #  this list will at the end all the dataframes created
    df_list = []

    df_b0_list = []    
    df_b1_list = []
    df_b2_list = []
    df_b3_list = []
    df_c3_list = []
    df_d3_list = []
    df_e3_list = []
    df_f1_list = []
    df_f2_list = []
    df_f3_list = []
    df_x1_list = []
    df_x2_list = []
    df_x_list = []


    # create b0: absolute differences and sums of a0   
    # this is not in the PRL. 
    for subset in itertools.combinations(col_a0, 2):
        if '+' in allowed_operations:
            cols = ['('+subset[0]+'+'+subset[1]+')']        
            data = df_a0[list(subset)].apply(_my_sum, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols))         
            
        if '-' in allowed_operations:
            cols = ['('+subset[0]+'-'+subset[1]+')']        
            data = df_a0[list(subset)].apply(_my_diff, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols))   
            
            cols = ['('+subset[1]+'-'+subset[0]+')']        
            data = df_a0[list(subset)].apply(_my_diff, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols))  
        
        if '|+|' in allowed_operations:
            cols = ['|'+subset[0]+'+'+subset[1]+'|']        
            data = df_a0[list(subset)].apply(_my_abs_sum, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols))     
        
        if '|-|' in allowed_operations:
            cols = ['|'+subset[0]+'-'+subset[1]+'|']        
            data = df_a0[list(subset)].apply(_my_abs_diff, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols))  
            
        if '/' in allowed_operations:
            cols = [subset[0]+'/'+subset[1]]        
            data = df_a0[list(subset)].apply(_my_div, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols))  

            cols = [subset[1]+'/'+subset[0]]        
            data = df_a0[list(subset)].apply(_my_div, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols))  

    
    # we kept itertools.combinations to make the code more uniform with the binary operations
    for subset in itertools.combinations(col_a0, 1):
        if '^2' in allowed_operations:
            cols = [subset[0]+'^2']        
            data = df_a0[list(subset)].apply(_my_power_2, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols))    
            
        if '^3' in allowed_operations:
            cols = [subset[0]+'^3']   
            data = df_a0[list(subset)].apply(_my_power_3, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols)) 

        if 'exp' in allowed_operations:
            cols = ['exp('+subset[0]+')']       
            data = df_a0[list(subset)].apply(_my_exp, axis=1)            
            df_b0_list.append(pd.DataFrame(data, columns=cols))        
        
        
    # create b1: absolute differences and sums of a1    
    for subset in itertools.combinations(col_a1, 2):
        if '+' in allowed_operations:
            cols = ['('+subset[0]+'+'+subset[1]+')']        
            data = df_a1[list(subset)].apply(_my_sum, axis=1)            
            df_b1_list.append(pd.DataFrame(data, columns=cols))         
            
        if '-' in allowed_operations:
            cols = ['('+subset[0]+'-'+subset[1]+')']        
            data = df_a1[list(subset)].apply(_my_diff, axis=1)            
            df_b1_list.append(pd.DataFrame(data, columns=cols))   

        if '|+|' in allowed_operations:
            cols = ['|'+subset[0]+'+'+subset[1]+'|']        
            data = df_a1[list(subset)].apply(_my_abs_sum, axis=1)            
            df_b1_list.append(pd.DataFrame(data, columns=cols))     
            
        if '|-|' in allowed_operations:
            cols = ['|'+subset[0]+'-'+subset[1]+'|']        
            data = df_a1[list(subset)].apply(_my_abs_diff, axis=1)            
            df_b1_list.append(pd.DataFrame(data, columns=cols))  

    # create b2: absolute differences and sums of a2    
    for subset in itertools.combinations(col_a2, 2):
        if '+' in allowed_operations:
            cols = ['('+subset[0]+'+'+subset[1]+')']        
            data = df_a2[list(subset)].apply(_my_sum, axis=1)            
            df_b2_list.append(pd.DataFrame(data, columns=cols))         
            
        if '-' in allowed_operations:
            cols = ['('+subset[0]+'-'+subset[1]+')']        
            data = df_a2[list(subset)].apply(_my_diff, axis=1)            
            df_b2_list.append(pd.DataFrame(data, columns=cols))   

        if '|+|' in allowed_operations:
            cols = ['|'+subset[0]+'+'+subset[1]+'|']        
            data = df_a2[list(subset)].apply(_my_abs_sum, axis=1)            
            df_b2_list.append(pd.DataFrame(data, columns=cols))         
            
        if '|-|' in allowed_operations:
            cols = ['|'+subset[0]+'-'+subset[1]+'|']        
            data = df_a2[list(subset)].apply(_my_abs_diff, axis=1)            
            df_b2_list.append(pd.DataFrame(data, columns=cols))   
 
    # create b3: absolute differences and sums of a3    
    for subset in itertools.combinations(col_a3, 2):
        if '+' in allowed_operations:
            cols = ['('+subset[0]+'+'+subset[1]+')']        
            data = df_a3[list(subset)].apply(_my_sum, axis=1)            
            df_b3_list.append(pd.DataFrame(data, columns=cols))         
            
        if '-' in allowed_operations:
            cols = ['('+subset[0]+'-'+subset[1]+')']        
            data = df_a3[list(subset)].apply(_my_diff, axis=1)            
            df_b3_list.append(pd.DataFrame(data, columns=cols))              

        if '|+|' in allowed_operations:
            cols = ['|'+subset[0]+'+'+subset[1]+'|']        
            data = df_a3[list(subset)].apply(_my_abs_sum, axis=1)            
            df_b3_list.append(pd.DataFrame(data, columns=cols))  
            
        if '|-|' in allowed_operations:
            cols = ['|'+subset[0]+'-'+subset[1]+'|']        
            data = df_a3[list(subset)].apply(_my_abs_diff, axis=1)            
            df_b3_list.append(pd.DataFrame(data, columns=cols))              

    # create c3: two steps:
    # 1) squares of a3 - unary operations 
    # we kept itertools.combinations to make the code more uniform with the binary operations
    for subset in itertools.combinations(col_a3, 1):
        if '^2' in allowed_operations:
            cols = [subset[0]+'^2']        
            data = df_a3[list(subset)].apply(_my_power_2, axis=1)            
            df_c3_list.append(pd.DataFrame(data, columns=cols))    
        if '^3' in allowed_operations:
            cols = [subset[0]+'^3']   
            data = df_a3[list(subset)].apply(_my_power_3, axis=1)            
            df_c3_list.append(pd.DataFrame(data, columns=cols)) 

            
    # 2) squares of b3 (only sums) --> sum squared of a3
    for subset in itertools.combinations(col_a3, 2):
        if '^2' in allowed_operations:
            cols = ['('+subset[0]+'+'+subset[1]+')^2']   
            data = df_a3[list(subset)].apply(_my_sum_power_2, axis=1)            
            df_c3_list.append(pd.DataFrame(data, columns=cols))        
            
        if '^3' in allowed_operations:
            cols = ['('+subset[0]+'+'+subset[1]+')^3']        
            data = df_a3[list(subset)].apply(_my_sum_power_3, axis=1)            
            df_c3_list.append(pd.DataFrame(data, columns=cols))

    # create d3: two steps:
    # 1) exponentials of a3 - unary operations 
    # we kept itertools.combinations to make the code more uniform with the binary operations
    for subset in itertools.combinations(col_a3, 1):
        if 'exp' in allowed_operations:
            cols = ['exp('+subset[0]+')']      
            df_subset = df_a3[list(subset)]
            data = df_subset.apply(_my_exp, axis=1)            
            df_d3_list.append(pd.DataFrame(data, columns=cols))    
            
    # 2) exponentials of b3 (only sums) --> exponential of sum of a3
    for subset in itertools.combinations(col_a3, 2):
        if 'exp' in allowed_operations:
            cols = ['exp('+subset[0]+'+'+subset[1]+')']    
            df_subset = df_a3[list(subset)]
            data = df_subset.apply(_my_sum_exp, axis=1)               
            df_d3_list.append(pd.DataFrame(data, columns=cols))        

    # create e3: two steps:
    # 1) exponentials of squared a3 - unary operations 
    # we kept itertools.combinations to make the code more uniform with the binary operations
    for subset in itertools.combinations(col_a3, 1):
        operations={'exp', '^2'}
        if operations <= set(allowed_operations):
            cols = ['exp('+subset[0]+'^2)']
            df_subset = df_a3[list(subset)]
            data = df_subset.apply(_my_exp_power_2, axis=1)            
            df_e3_list.append(pd.DataFrame(data, columns=cols))    
            
        operations={'exp', '^3'}
        if operations <= set(allowed_operations):
            try:
                cols = ['exp('+subset[0]+'^3)']
                df_subset = df_a3[list(subset)]
                data = df_subset.apply(_my_exp_power_3, axis=1)            
                df_e3_list.append(pd.DataFrame(data, columns=cols)) 
            except OverflowError as e:
                print('Dropping feature combination that caused under/overflow.\n')

            
    # 2) exponentials of b3 (only sums) --> exponential of sum of a3
    for subset in itertools.combinations(col_a3, 2):
        operations={'exp', '^2'}
        if operations <= set(allowed_operations):
            cols = ['exp(('+subset[0]+'+'+subset[1]+')^2)']
            df_subset = df_a3[list(subset)]
            data = df_subset.apply(_my_sum_exp_power_2, axis=1)            
            df_e3_list.append(pd.DataFrame(data, columns=cols))        

        operations={'exp', '^3'}
        if operations <= set(allowed_operations):
            try:
                cols = ['exp(('+subset[0]+'+'+subset[1]+')^3)']
                df_subset = df_a3[list(subset)]
                data = df_subset.apply(_my_sum_exp_power_3, axis=1)            
                df_e3_list.append(pd.DataFrame(data, columns=cols))   
            except OverflowError as e:
                print('Dropping feature combination that caused under/overflow.\n')

    # make dataframes from lists, check if they are not empty
    # we make there here because they are going to be used to further
    # combine the features
    if not df_a0.empty: 
        df_list.append(df_a0)
        
    if not df_a1.empty: 
        df_x1_list.append(df_a1)
        df_list.append(df_a1)

    if not df_a2.empty: 
        df_x1_list.append(df_a2)
        df_list.append(df_a2)
        
    if not df_a3.empty: 
        df_x1_list.append(df_a3)
        df_list.append(df_a3)



    if df_b0_list: 
        df_b0 = pd.concat(df_b0_list, axis=1)
        col_b0 = df_b0.columns.tolist()
        df_b0.to_csv('./df_b0.csv', index=True)
        df_list.append(df_b0)
        
    if df_b1_list: 
        df_b1 = pd.concat(df_b1_list, axis=1)
        col_b1 = df_b1.columns.tolist()
        df_x1_list.append(df_b1)
        df_list.append(df_b1)

    if df_b2_list: 
        df_b2 = pd.concat(df_b2_list, axis=1)
        col_b2 = df_b2.columns.tolist()
        df_x1_list.append(df_b2)
        df_list.append(df_b2)
        
    if df_b3_list: 
        df_b3 = pd.concat(df_b3_list, axis=1)
        col_b3 = df_b3.columns.tolist()        
        df_x1_list.append(df_b3)
        df_list.append(df_b3)
    
    if df_c3_list:
        df_c3 = pd.concat(df_c3_list, axis=1)
        col_c3 = df_c3.columns.tolist()
        df_x2_list.append(df_c3)
        df_list.append(df_c3)

    if df_d3_list:
        df_d3 = pd.concat(df_d3_list, axis=1)
        col_d3 = df_d3.columns.tolist()
        df_x2_list.append(df_d3)
        df_list.append(df_d3)

    if df_e3_list:
        df_e3 = pd.concat(df_e3_list, axis=1)
        col_e3 = df_e3.columns.tolist()
        df_x2_list.append(df_e3)
        df_list.append(df_e3)

    if df_x1_list:
        df_x1 = pd.concat(df_x1_list, axis=1)
        col_x1 = df_x1.columns.tolist()
                
    if df_x2_list:
        df_x2 = pd.concat(df_x2_list, axis=1)
        col_x2 = df_x2.columns.tolist()

    if df_x1_list and df_x2_list:
        for el_x1 in col_x1:
            for el_x2 in col_x2:
                if '/' in allowed_operations:
                    cols = [el_x1+'/'+el_x2] 
                    #now the operation is between two dataframes
                    data = df_x1[el_x1].divide(df_x2[el_x2])     
                    df_x_list.append(pd.DataFrame(data, columns=cols))   
     

    if df_f1_list:
        df_f1 = pd.concat(df_f1_list, axis=1)
        col_f1 = df_f1.columns.tolist()
        df_list.append(df_f1)

                
    if df_x_list:
        df_x = pd.concat(df_x_list, axis=1)
        col_x = df_x.columns.tolist()
        df_list.append(df_x)




    if df_list:
        df_combined_features = pd.concat(df_list, axis=1)
    elif is_print:
        print('No features selected. Please select at least two primary features.')
        

    if is_print:
        print('Number of total features generated: {0}'.format(df_combined_features.shape[1]))
    
    return df_combined_features

