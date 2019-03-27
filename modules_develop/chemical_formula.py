from ase.data import chemical_symbols
import numpy as np
import re

exchange_list = ['FLi', 'ClLi', 'BrLi', 'ILi', 'AsB', 'FNa', 'ClNa', 'BrNa', 'INa', 'CSi', 'FK', 'ClK', 'BrK', 'IK', 'ClCu', 'BrCu', 'OZn', 'SZn', 'SeZn', 'TeZn', 'AsGe', 'FRb', 'ClRb', 'BrRb', 'IRb', 'OSr', 'SSr', 'SeSr', 'AsIn', 'ClCs', 'BrCs', 'SiSn', 'CGe', 'GeSn', 'CSn']

exchange_dict = {mat: "".join(re.findall('[A-Z][^A-Z]*', mat)[::-1]) for mat in exchange_list}
    
def get_chemical_formula_binaries(atoms):
    """Function that inverts chemical formula AB to BA if AB is in exchange_list.
    Input can be ase atoms object or string."""
    try:
        formula = atoms.get_chemical_formula()
    except:
        formula = atoms

    try:
        return exchange_dict[formula]
    except:
        return formula

