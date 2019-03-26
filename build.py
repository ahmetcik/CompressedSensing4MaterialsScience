from modules_build.json_lists import json_dict
from modules_build.nomad_structures import NOMADStructure
import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
import pandas as pd
from modules_build.chemical_formula import get_chemical_formula_binaries

# make out of json files atoms_structure list
# for developer
Structures_list = ['ZB', 'RS', 'CsCl', 'NiAs', 'CrB'][:2]

#spacegroup_tuples = [(225, 221), (216, 227)] # [(RS), (ZB)]
for struc in Structures_list:
    Atoms_list = []
    
    for json_path in json_dict[struc]:
        nomad_struc = NOMADStructure(in_file=json_path, file_format='NOMAD', take_first='False')
        atoms_ase = nomad_struc.atoms[0,0]
        energy    = nomad_struc.energy_total__eV[0,0]
        atoms_ase.set_calculator(SinglePointCalculator(atoms_ase, energy=energy))

        Atoms_list.append(atoms_ase)
    
    ase.io.write('data_build/%s.xyz' % struc, Atoms_list)

structures_RS = ase.io.read('data_build/RS.xyz', ':')
structures_ZB = ase.io.read('data_build/ZB.xyz', ':')

structures_dict_RS = {get_chemical_formula_binaries(at): at  for at in structures_RS}
structures_dict_ZB = {get_chemical_formula_binaries(at): at  for at in structures_ZB}
energies_dict_RS = {get_chemical_formula_binaries(at): at.get_potential_energy()/len(at)  for at in structures_RS}
energies_dict_ZB = {get_chemical_formula_binaries(at): at.get_potential_energy()/len(at)  for at in structures_ZB}

# create and print data frame of energies and energy differences
df = pd.DataFrame([energies_dict_RS, energies_dict_ZB, structures_dict_RS, structures_dict_ZB], 
                           index=['RS', 'ZB', 'struc_obj_RS', 'struc_obj_ZB']).T
df['energy_diff'] = df['RS'] - df['ZB']
df['min_struc_type'] = df[['RS', 'ZB']].idxmin(axis=1)
df.index.name = 'chemical_formula'

# get min ase structures
struc_obj_min_labels = ('struc_obj_'+df['min_struc_type'] ).values
struc_obj_min = [df.loc[df.index[i], struc_obj_min_labels[i]] for i in range(df.shape[0])]
df['struc_obj_min'] =  struc_obj_min

df_atomic = pd.read_csv('data_build/Octet_binaries_atomic_features.csv', index_col=0)

df = df.merge(df_atomic,  left_index=True, right_index=True)

df = df.rename(index=str, columns={'RS': 'energy_RS', 'ZB': 'energy_ZB'})
columns  = ['energy_RS', 'energy_ZB', 'energy_diff', 'min_struc_type'][:]
columns += df_atomic.columns.tolist() + ['struc_obj_RS', 'struc_obj_ZB', 'struc_obj_min']
df[columns].to_pickle('data/data.pkl')
