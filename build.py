import pandas as pd

# create thumbnail pngs
Structures_list = ['ZB', 'RS']
df = pd.read_pickle("data/data.pkl")
chemical_formulas = df.index.tolist()
for struc in Structures_list:
    atoms_list = df['struc_obj_%s' % struc].tolist()
    for i, atoms in enumerate(atoms_list):
        chemical_formula = chemical_formulas[i]
        
        # create supercell
        atoms = atoms * [3, 3, 3]

        # create png out of supercell
        atoms.write("data/Thumbnail_%s_%s.png" %(struc, chemical_formula), format='png', rotation='10z,-80x', radii=0.5, scale=100)


