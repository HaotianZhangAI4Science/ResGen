from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import glob
import os.path as osp
from rdkit.Chem import rdMolAlign as MA
import copy

def GetBestRMSD(probe, ref):
    probe = Chem.RemoveHs(probe)
    ref = Chem.RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd
def read_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)
    return data

def read_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    mols = [i for i in supp]
    return mols
def write_pkl(data_list, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_list, f)

def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = copy.deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
    for confId in confIds:
        AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
        rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
        rmsd_list.append(rmsd)
    # mol3d = Chem.RemoveHs(mol3d)
    rmsd_list = np.array(rmsd_list)
    return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]

dirs = glob.glob('./resgen/*')
def conf_analysis(i):
    dir_ = dirs[i]
    all_RMSD = []
    for mol_index in range(1,200):
        try:
            gen_mols = read_sdf(osp.join(dir_, 'SDF', f'{mol_index}.sdf'))
            # for conf_idx in range(len(docked_mols)):
            #     try:
            #         rmsd = CalcRMS(gen_mols[0], docked_mols[conf_idx])**0.5
            #         all_RMSD.append(rmsd)
            #     except:
            #         ...
            #         #print('failed{}'.format(mol_index))
            rmsd_max, rmsd_min, rmsd_med = get_rdkit_rmsd(gen_mols[0])
            all_RMSD.append(rmsd_min)
        except:
            ...
    return all_RMSD
    
from tqdm import tqdm
all_rmsd = []
for i in tqdm(range(len(dirs))):
    try:
        rmsd = conf_analysis(i)
        all_rmsd.extend(rmsd)
    except:
        ...
write_pkl(all_rmsd, 'resgen_conf_minRMSD.pkl')