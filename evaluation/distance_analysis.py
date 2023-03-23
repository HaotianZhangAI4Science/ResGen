import os.path as osp
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
from rdkit import Chem
def read_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(data_list, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_list, f)

def read_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file)
    mols_list = [i for i in supp]
    return mols_list

#compute KL Divergence
"""KL Divergence(P|Q)"""
def KL_div(p_probs, q_probs):    
    KL_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(KL_div)
def JS_Div(p, q, normalize=False):
    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    if normalize:
        p /= p.sum()
        q /= q.sum()
    m = (p + q) / 2
    return (KL_div(p, m) + KL_div(q, m)) / 2
# JS Divergence is symmetric

def pair_extract(mols):
    '''
    input: mols list
    return: a dict which keys are atomic type and values are distances 
    '''
    pair_dict = {}

    for mol in mols:
        conf = mol.GetConformer().GetPositions()
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            begin_type = mol.GetAtomWithIdx(begin_idx).GetSymbol()
            end_type = mol.GetAtomWithIdx(end_idx).GetSymbol()
            begin_pos = conf[begin_idx]
            end_pos = conf[end_idx]
            dis = np.around(np.linalg.norm(begin_pos - end_pos),8)
            try:
                pair_dict[begin_type + end_type].append(dis)
                pair_dict[end_type + begin_type].append(dis)
            except:
                pair_dict[begin_type + end_type] = []
                pair_dict[end_type + begin_type] = []
                pair_dict[begin_type + end_type].append(dis)
                pair_dict[end_type + begin_type].append(dis)

    return pair_dict

keys = ['CC','CN','CO','NO','SO','CCl','CF']

            