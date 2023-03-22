# python process_data.py --raw_data ./data/crossdocked_pocket10 
import os
import argparse
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import lmdb
import pickle
class self:
    raw_path = '/home/haotian/Molecule_Generation/Res2Mol/data/crossdocked_pocket10'
self.index_path = os.path.join(self.raw_path, 'index.pkl')

from utils.feats import parse_sdf_file, parse_PDB_v2
from utils.data import torchify_dict, ProteinLigandData
from utils.datasets import *
from utils.transforms import *
from utils.train import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/train_res.yml')
parser.add_argument('--raw_data', type=str, default='./data/crossdocked_pocket10')
args = parser.parse_args()
config = load_config(args.config)

index_path = os.path.join(args.raw_data, 'index.pkl')
processed_path = os.path.join(os.path.dirname(args.raw_data), os.path.basename(args.raw_data) + '_processed.lmdb')
name2id_path = os.path.join(os.path.dirname(args.raw_data), os.path.basename(args.raw_data) + '_processed_name2id.pt')
db = lmdb.open(
    processed_path,
    map_size=15*(1024*1024*1024),   # 15GB
    create=True,
    subdir=False,
    readonly=False, # Writable
)
with open(index_path, 'rb') as f:
    index = pickle.load(f)

num_skipped = 0
with db.begin(write=True, buffers=True) as txn:
    for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(index):
        if pocket_fn is None: continue
        try:
            pocket_dict = parse_PDB_v2(os.path.join(args.raw_data, pocket_fn))
            ligand_dict = parse_sdf_file(os.path.join(args.raw_data, ligand_fn))
            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=torchify_dict(pocket_dict),
                ligand_dict=torchify_dict(ligand_dict),
            )
            data.protein_filename = pocket_fn
            data.ligand_filename = ligand_fn
            txn.put(
                key = str(i).encode(),
                value = pickle.dumps(data)
            )
        except:
            num_skipped += 1
            if num_skipped % 1000 == 0:
                print('skipping {}, and the current idx is {}'.format(num_skipped, i))
            #print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
            continue
db.close()

print('done, the total skipping mol is {}'.format(num_skipped))

ligand_featurizer = FeaturizeLigandAtom()
masking = get_mask(config.train.transform.mask)
composer = Res2AtomComposer(27, ligand_featurizer.feature_dim, config.model.encoder.knn)
edge_sampler = EdgeSample(config.train.transform.edgesampler)  #{'k': 8}
cfg_ctr = config.train.transform.contrastive
contrastive_sampler = ContrastiveSample(cfg_ctr.num_real, cfg_ctr.num_fake, cfg_ctr.pos_real_std, cfg_ctr.pos_fake_std, config.model.field.knn)
transform = Compose([
    RefineData(),
    LigandCountNeighbors(),
    ligand_featurizer,
    masking,
    composer,

    FocalBuilder(),
    edge_sampler,
    contrastive_sampler,
])

db = lmdb.open(
            processed_path,
            map_size=5*(1024*1024*1024),   # 5GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
with db.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))

name2id = {}
for i in tqdm(range(len(keys)), 'Indexing'):
    try:
        key = keys[i]
        data = pickle.loads(db.begin().get(key))
        data.id = i
        assert data.pkt_node_xyz.size(0) > 0
        transform(data)
    except AssertionError as e:
        print(i,e)
        continue
    name = (data.protein_filename, data.ligand_filename)
    name2id[name] = i
torch.save(name2id, name2id_path)