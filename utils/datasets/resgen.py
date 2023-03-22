import lmdb
import pickle
from torch.utils.data import Dataset
import os
import torch

class ResGenDataset(Dataset):

    def __init__(self, raw_path='./data/crossdocked_pocket10', transform=None):
        super().__init__()
        self.raw_path = raw_path
        self.processed_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_processed.lmdb')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.transform = transform
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path), os.path.basename(self.raw_path) + '_name2id.pt')
        self.keys = None
        if not os.path.exists(self.processed_path):
            raise Exception('Please processing the data first!')  
        self.name2id = torch.load(self.name2id_path)
    
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
        
    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)
    
    def __getitem__(self, idx):
        
        self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        assert data.pkt_node_xyz.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data