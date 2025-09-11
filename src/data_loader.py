from typing import Optional, Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from collections import defaultdict
from cppimport import imp_from_filepath
import tracemalloc
import warnings


UNIFRORM_SAMPLER_PATH = "src/unifrom_sampling.cpp"
UNIFORM_SP_MODULE = imp_from_filepath(UNIFRORM_SAMPLER_PATH)
uniform_sampler:Callable = UNIFORM_SP_MODULE.uniform_sampler
SEED_VAL = 51
seed_fucn:Callable = UNIFORM_SP_MODULE.seed
seed_fucn(SEED_VAL)

class RecSysDataset(Dataset):
    def __init__(self,
                 data_path:Path|str,
                 is_train_data:bool = True,
                 add_self_loop: bool = False,
                 neg_per_pos:Optional[int]=None) -> None:
        
        if neg_per_pos is None:
            raise ValueError("neg_per_pos must be set for training data")
        
        self.data_path = data_path
        self.is_train_data = is_train_data
        self.add_self_loop = add_self_loop
        self.neg_per_pos = neg_per_pos

        self.users_l, self.items_l = self._load_from_txt(data_path) # ex. [user1, user1, user1, user2,...], [item10, item56, item78, item44...]
        self.num_users = max(self.users_l) + 1 # assuming user ids are 0-indexed
        self.num_items = max(self.items_l) + 1 # assuming item ids are 0-indexed
        self.data_size = len(self.users_l) # number of all interactions

        if is_train_data:
            print(f"Train data size: {self.data_size}")
            self.users_l, self.items_l, self.user_pos_items_dict_train = self._load_from_txt(data_path, build_dict=True)
        else:
            print(f"Test data size: {self.data_size}")
            self.users_l, self.items_l = self._load_from_txt(data_path)
            self.prepare_test_data()
            del self.users_l
            del self.items_l

    @staticmethod
    def _load_from_txt(path, build_dict: bool = False):
        users_l = []
        items_l = []
        user_pos_dict = defaultdict(list) if build_dict else None

        with open(path) as f:
            for line in f:
                if line.strip():
                    parts = line.split()
                    uid = int(parts[0])
                    items = map(int, parts[1:])
                    for item in items:
                        users_l.append(uid)
                        items_l.append(item)
                        if build_dict:
                            user_pos_dict[uid].append(item)

        if build_dict:
            return users_l, items_l, user_pos_dict
        return users_l, items_l

    
    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
    

    @staticmethod
    def _get_user_item_matrix(users:list[int],
                              items:list[int],
                              n_users:int,
                              n_items:int) -> csr_matrix:
        
        user_item_matrix_train = csr_matrix((np.ones(len(users)), (users, items)),
                                      shape=(n_users, n_items))
        return user_item_matrix_train
    

    @staticmethod
    def _get_adjacency_matrix(user_item_matrix_train:csr_matrix,
                              add_self_loop:bool = False):
        
        adj_mat = sp.bmat([[None, user_item_matrix_train],
                           [user_item_matrix_train.transpose(), None]],
                           format='csr',
                           dtype=np.float32)

        if add_self_loop:
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0], dtype=np.float32)
        
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        return norm_adj
    
    def get_graph(self) -> torch.sparse.FloatTensor:
        user_item_matrix_train = self._get_user_item_matrix(users=self.users_l,
                                                            items=self.items_l,
                                                            n_users=self.num_users,
                                                            n_items=self.num_items)
                
        norm_adj = self._get_adjacency_matrix(user_item_matrix_train=user_item_matrix_train,
                                              add_self_loop=self.add_self_loop)
                
        graph = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce()
        return graph
    
    def prepare_train_data(self) -> None:
            # uniformly sample triplets from the user_pos_items_dict_train using C++ implementation
            data = uniform_sampler(data_size=self.data_size,
                                   users_count=self.num_users,
                                   user_pos_items_dict_train=self.user_pos_items_dict_train,
                                   items_count=self.num_items,
                                   neg_per_pos = self.neg_per_pos)

            self.data = torch.tensor(data, dtype=torch.long)

    def prepare_test_data(self) -> None:
        unique_users, inverse_idx = np.unique(self.users_l, return_inverse=True)
        counts = np.bincount(inverse_idx)
        max_len = counts.max() + 1  
        arr = np.full((len(unique_users), max_len), 0, dtype=int)
        arr[:, 0] = unique_users

        offsets = np.zeros(len(unique_users), dtype=int)  
        for idx, u in enumerate(inverse_idx):
            pos = offsets[u] + 1  # +1 because column 0 is user id
            arr[u, pos] = self.items_l[idx]
            offsets[u] += 1
        self.data = torch.from_numpy(arr)

    def __len__(self)-> int:
        return len(self.data)
    
    def __getitem__(self, index:int):
        return self.data[index]
    
if __name__ == "__main__":
    # Do debugging prints here 
    dataset = RecSysDataset(data_path="/Users/aregpetrosyan/Desktop/areg/LightGCN/data/processed/train.txt",
                           is_train_data=True,
                           add_self_loop=False,
                           neg_per_pos=4)
    train_data = dataset.prepare_train_data()
    test_data = dataset.get_test_data()
    graph = dataset.get_graph()
