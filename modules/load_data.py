import ase.io
import pandas as pd
def load_data():
    df = pd.read_csv("data/data_table.csv", index_col=0)
    df['struc_obj_RS'] = ase.io.read('data/structures_rs.xyz', ':')
    df['struc_obj_ZB'] = ase.io.read('data/structures_zb.xyz', ':')
    
    struc_obj_min_labels = ('struc_obj_'+df['min_struc_type'] ).values
    df['struc_obj_min'] =  [df.loc[df.index[i], struc_obj_min_labels[i]] for i in range(df.shape[0])]
    return df
