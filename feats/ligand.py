from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem.rdchem import BondType
import torch

#这个相当于先把atomic_number 转换为 one_hot
atomic_num_to_type = {5:0, 6:1, 7:2, 8:3, 9:4, 12:5, 13:6, 14:7, 15:8, 16:9, 17:10, 21:11, 23:12, 26:13, 29:14, \
    30:15, 33:16, 34:17, 35:18, 39:19, 42:20, 44:21, 45:22, 51:23, 53:24, 74:25, 79:26}
atomic_element_to_type = {'C':27, 'N':28, 'O':29, 'NA':30, 'MG':31, 'P':32, 'S':33, 'CL':34, 'K':35, \
    'CA':36, 'MN':37, 'CO':38, 'CU':39, 'ZN':40, 'SE':41, 'CD':42, 'I':43, 'CS':44, 'HG':45}
bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC:1.5}

def read_lig_sdf(sdf_file):
    '''
    mol_src: the path of a .sdf file
    return: rdkit.Chem.rdmolfiles.SDMolSupplier
    '''
    supp = Chem.SDMolSupplier()
    supp.SetData(open(sdf_file).read(), removeHs=False, sanitize=False)
    return supp

def extract_lig_sdf(lig_supplier):
    lig_mol = Chem.rdmolops.RemoveAllHs(lig_supplier[0], sanitize=False)
    lig_n_atoms = lig_mol.GetNumAtoms()
    lig_pos = lig_supplier.GetItemText(0).split('\n')[4:4+lig_n_atoms]
    lig_position = np.array([[float(x) for x in line.split()[:3]] for line in lig_pos], dtype=np.float32)
    lig_atom_type = np.array([atomic_num_to_type[atom.GetAtomicNum()] for atom in lig_mol.GetAtoms()])
    lig_con_mat = np.zeros([lig_n_atoms, lig_n_atoms], dtype=int)
    for bond in lig_mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond_to_type[bond.GetBondType()]
        lig_con_mat[start, end] = bond_type
        lig_con_mat[end, start] = bond_type
    lig_atom_type = torch.tensor(lig_atom_type) #[lig_n_atoms]
    lig_position = torch.tensor(lig_position)  #[lig_n_atoms, 3]
    lig_atom_bond_valency = torch.tensor(np.sum(lig_con_mat, axis=1)) #[lig_n_atoms]
    lig_con_mat = torch.tensor(lig_con_mat) #[lig_n_atoms, lig_n_atoms]

    return lig_position, lig_atom_type, lig_con_mat, lig_atom_bond_valency