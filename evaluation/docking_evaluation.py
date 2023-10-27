import sys
import os
from glob import glob
import os.path as osp
import numpy as np
from docking_utils import prepare_target, mol2centroid, docking_with_sdf, get_result, prepare_ligand, sdf2centroid, scoring_with_sdf
from tqdm import tqdm
import pickle
import shutil
import torch
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
def read_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(data_list, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_list, f)


files = glob('/home/haotian/molecules_confs/Protein_test/ResGen/genereated/*')
for i in range(len(files)):
    if osp.exists(files[i]+'/gen_docking_results.pkl'):
        continue
    try:
        work_dir = files[i]+'/SDF'
        protein_file = glob(files[i]+'/SDF/*.pdb')[0]
        ori_lig_file = max(glob(files[i]+'/*.sdf'), key=len, default='')
        protein_filename = protein_file.split('/')[-1]

        prepare_target(work_dir, protein_filename, verbose=0)
        protein_pdbqt = protein_filename +'qt'
        centroid = sdf2centroid(ori_lig_file)
        result_list = []

        for j in tqdm(range(1, 200)):
            try:
                lig_pdbqt = prepare_ligand(work_dir, f'{j}.sdf', verbose=1)
                docked_sdf = docking_with_sdf(work_dir,protein_pdbqt, lig_pdbqt, centroid, verbose=0)
                result = get_result(docked_sdf=docked_sdf, ref_mol= None)
                result_list.append(result)
            except:
                ...

        write_pkl(result_list, files[i]+'/gen_docking_results.pkl')
    except:
        ...


        