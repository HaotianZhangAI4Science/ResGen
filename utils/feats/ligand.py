import torch
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

atomic_num_to_type = {5:0, 6:1, 7:2, 8:3, 9:4, 12:5, 13:6, 14:7, 15:8, 16:9, 17:10, 21:11, 23:12, 26:13, 29:14, \
    30:15, 33:16, 34:17, 35:18, 39:19, 42:20, 44:21, 45:22, 51:23, 53:24, 74:25, 79:26}
atomic_element_to_type = {'C':27, 'N':28, 'O':29, 'NA':30, 'MG':31, 'P':32, 'S':33, 'CL':34, 'K':35, \
    'CA':36, 'MN':37, 'CO':38, 'CU':39, 'ZN':40, 'SE':41, 'CD':42, 'I':43, 'CS':44, 'HG':45}
bond_to_type = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3, BondType.AROMATIC:1.5}

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}


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


def parse_sdf_file(path):

    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=False)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    with open(path, 'r') as f:
        sdf = f.read()

    sdf = sdf.splitlines()
    num_atoms, num_bonds = map(int, [sdf[3][0:3], sdf[3][3:6]])
    assert num_atoms == rd_num_atoms

    ptable = Chem.GetPeriodicTable()

    element, pos = [], []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    for atom_line in map(lambda x:x.split(), sdf[4:4+num_atoms]):
        x, y, z = map(float, atom_line[:3])
        symb = atom_line[3]
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())
        element.append(atomic_number)
        pos.append([x, y, z])
        
        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight

    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)

    element = np.array(element, dtype=np.int)
    pos = np.array(pos, dtype=np.float32)

    BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
    bond_type_map = {
        1: BOND_TYPES[BondType.SINGLE],
        2: BOND_TYPES[BondType.DOUBLE],
        3: BOND_TYPES[BondType.TRIPLE],
        4: BOND_TYPES[BondType.AROMATIC],
    }
    row, col, edge_type = [], [], []
    for bond_line in sdf[4+num_atoms:4+num_atoms+num_bonds]:
        start, end = int(bond_line[0:3])-1, int(bond_line[3:6])-1
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_type_map[int(bond_line[6:9])]]

    edge_index = np.array([row, col], dtype=np.long)
    edge_type = np.array(edge_type, dtype=np.long)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
    }
    return data

