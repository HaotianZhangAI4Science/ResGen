import os
import argparse
import warnings
import os.path as osp
from easydict import EasyDict
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from rdkit import Chem
import numpy as np
import torch
import sys
from copy import deepcopy
sys.path.append('../../Res2Mol')
from utils.feats.protein import get_protein_feature_v2
from Bio.PDB import NeighborSearch, Selection
from utils.protein_ligand import parse_sdf_file
from utils.data import torchify_dict, ProteinLigandData
#from feats.protein import 
from tqdm.auto import tqdm
from models.ResGen import ResGen
from utils.transforms import *
from utils.misc import load_config, transform_data
from utils.reconstruct import *
from utils.datasets.res2mol import Res2MolDataset
from utils.sample import get_init, get_next, logp_to_rank_prob
from utils.sample import STATUS_FINISHED, STATUS_RUNNING

def read_sdf(file):
    supp = Chem.SDMolSupplier(file)
    return [i for i in supp]

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import warnings

def pdb_to_pocket_data(pdb_file, bbox_size=10.0, mol_file=None, center=None):
    '''
    use the sdf_file as the center 
    '''
    if mol_file is not None:
        prefix = mol_file.split('.')[-1]
        if prefix == 'mol2':
            center = Chem.MolFromMol2File(mol_file, sanitize=False).GetConformer().GetPositions()
            center = np.array(center)
        elif prefix == 'sdf':
            supp = Chem.SDMolSupplier(mol_file, sanitize=False)
            center = supp[0].GetConformer().GetPositions()
        else:
            print('The File type of Molecule is not support')
    elif center is not None:
        center = center
    else:
        print('You must specify the original ligand file or center')
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('target', pdb_file)[0]
    atoms  = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atoms)
    close_residues= []
    dist_threshold = bbox_size
    for a in center:  
        close_residues.extend(ns.search(a, dist_threshold, level='R'))
    close_residues = Selection.uniqueify(close_residues)
    protein_dict = get_protein_feature_v2(close_residues)

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict = protein_dict,
        ligand_dict = {
            'element': torch.empty([0,], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0,], dtype=torch.long),
        }
    )
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/sample.yml')
parser.add_argument('--outdir', type=str, default='../outputs')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--data_dir', type=str,default='../data/crossdocked_pocket10' )
parser.add_argument('--check_point',type=str,default='../logs/use/57.pt')
#args = parser.parse_args(['--data_id','2'])
args = parser.parse_args([])
config = load_config(args.config)
# define the model and transform function (process the data again)
contrastive_sampler = ContrastiveSample()
ligand_featurizer = FeaturizeLigandAtom()
transform = Compose([
    RefineData(),
    LigandCountNeighbors(),
    ligand_featurizer
])

ckpt = torch.load(args.check_point, map_location=args.device)
model = ResGen(
    ckpt['config'].model, 
    num_classes = contrastive_sampler.num_elements,
    num_bond_types = 3,
    protein_res_feature_dim = (27,3),
    ligand_atom_feature_dim = (13,1),
).to(args.device)
print('Num of parameters is {0:.4}M'.format(np.sum([p.numel() for p in model.parameters()]) /100000 ))
model.load_state_dict(ckpt['model'])

pdb_file = './14gs_pocket.pdb'
mol= read_sdf('./14gs_ligand.sdf')[0]
atomCoords = mol.GetConformers()[0].GetPositions()
data = pdb_to_pocket_data(pdb_file, center=atomCoords, bbox_size=10)

mask = LigandMaskAll()
composer = Res2AtomComposer(27, ligand_featurizer.feature_dim, ckpt['config'].model.encoder.knn)
masking = Compose([
    mask, 
    composer
])
data = transform(data)
data = transform_data(data, masking)

np.seterr(invalid='ignore') 
pool = EasyDict({
    'queue': [],
    'failed': [],
    'finished': [],
    'duplicate': [],
    'smiles': set(),
})

data = transform_data(deepcopy(data), masking)
init_data_list = get_init(data.to(args.device),   # sample the initial atoms
        model = model,
        transform=composer,
        threshold=config.sample.threshold
)
pool.queue = init_data_list
#rint('Start to generate novel molecules with 3D conformation located in the protein pocket!')
#print('The protein pocket is {}, init length is {}'.format(data.protein_filename, len(init_data_list)))
global_step = 0 
while len(pool.finished) < config.sample.num_samples:
    global_step += 1
    if global_step > config.sample.max_steps:
        break
    queue_size = len(pool.queue)
    # # sample candidate new mols from each parent mol
    queue_tmp = []
    for data in pool.queue:
        nexts = []
        data_next_list = get_next(
            data.to(args.device), 
            model = model,
            transform = composer,
            threshold = config.sample.threshold
        )

        for data_next in data_next_list:
            if data_next.status == STATUS_FINISHED:
                try:
                    rdmol = reconstruct_from_generated_with_edges(data_next)
                    data_next.rdmol = rdmol
                    mol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
                    smiles = Chem.MolToSmiles(mol)
                    data_next.smiles = smiles
                    if smiles in pool.smiles:
                        #print('Duplicate molecule: %s' % smiles)
                        pool.duplicate.append(data_next)
                    elif '.' in smiles:
                        #print('Failed molecule: %s' % smiles)
                        pool.failed.append(data_next)
                    else:   # Pass checks
                        #print('Success: %s' % smiles)
                        pool.finished.append(data_next)
                        pool.smiles.add(smiles)
                except MolReconsError:
                    #print('Reconstruction error encountered.')
                    pool.failed.append(data_next)
            elif data_next.status == STATUS_RUNNING:
                nexts.append(data_next)

        queue_tmp += nexts
    prob = logp_to_rank_prob(np.array([p.average_logp[2:] for p in queue_tmp]),)  # (logp_focal, logpdf_pos), logp_element, logp_hasatom, logp_bond
    n_tmp = len(queue_tmp)
    if n_tmp == 0:
        print('This Generation has filures!')
        break
    else:
        next_idx = np.random.choice(np.arange(n_tmp), p=prob, size=min(config.sample.beam_size, n_tmp), replace=False)
    pool.queue = [queue_tmp[idx] for idx in next_idx]

sdf_dir = './'
sdf_file = os.path.join(sdf_dir,'test_gen.sdf')
writer = Chem.SDWriter(sdf_file)
for j in range(len(pool['finished'])):
    writer.write(pool['finished'][j].rdmol)
#print('{}th has been generated, {} saved'.format(i, sdf_file))
writer.close()

SDF_dir = './SDF'
os.makedirs(SDF_dir, exist_ok=True)
for j in range(len(pool['finished'])):
    writer = Chem.SDWriter(SDF_dir+f'/{j}.sdf')
    writer.write(pool['finished'][j].rdmol)
    writer.close()