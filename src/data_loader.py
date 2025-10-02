from typing import Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from collections import defaultdict
from cppimport import imp_from_filepath
import sys
from typing_extensions import override


sys.path.append("src")
UNIFRORM_SAMPLER_PATH = "src/unifrom_sampling.cpp"
UNIFORM_SP_MODULE = imp_from_filepath(UNIFRORM_SAMPLER_PATH)
uniform_sampler:Callable = UNIFORM_SP_MODULE.uniform_sampler
SEED_VAL = 51
seed_fucn:Callable = UNIFORM_SP_MODULE.seed
seed_fucn(SEED_VAL)

class BaseUI_Interation_Dataset(Dataset):

    ''' Base class for datasets with user-item interactions.
    Loads user-item interactions from a text file.
    Each line in the file should be formatted as:
    user_id item_id1 item_id2 ...
    Assumes user and item IDs are 0-indexed integers.
    Attributes:
    - num_users: Total number of unique users.
    - num_items: Total number of unique items.
    - data_size: Total number of interactions.
    - data: (in train mode will be [user_id, pos_item_id1, neg_sampled_id1,...]
            (in test mode will be [user_id, pos_item_id1, pos_item_id2,...])
    '''

    def __init__(self,
                 data_path: Path | str,
                 is_train_data: bool = True,
                 neg_per_pos: int | None = None) -> None:

        if is_train_data and (neg_per_pos is None or neg_per_pos <= 0):
            raise ValueError("neg_per_pos must be a positive integer for training data.")

        self.dump_dir = Path("data/dump")
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path
        self.is_train_data = is_train_data
        self.neg_per_pos = neg_per_pos

        # Load interactions
        if is_train_data:
            self.users_l, self.items_l, self.user_pos_dict = self._load_from_txt(data_path, build_dict=True)

            self.num_users = max(self.users_l) + 1  
            self.num_items = max(self.items_l) + 1  
            self.data_size = len(self.users_l)
            print(f"Train data size: {self.data_size}")
            print(f"Number of users in train: {self.num_users}, Number of items in train: {self.num_items}")
            self.prepare_train_data()
            # self.data.shape = [num_interactions, 1 (userid) + 1 (pos_inter) + neg_per_pos]
        else:
            users_l, items_l = self._load_from_txt(data_path,build_dict=False)
            self.prepare_test_data(users_l, items_l)
            self.num_users = max(users_l) + 1  
            self.num_items = max(items_l) + 1  
            self.data_size = len(users_l)
            print(f"Test data size: {self.data_size}")
            print(f"Number of users in test: {self.num_users}, Number of items in test: {self.num_items}")
            max_interactions_count_in_test = self.data.shape[1]
            print(f"Max interactions count in test data for a user: {max_interactions_count_in_test - 1}")
            # self.data.shape = [num_unique_users, 1 (userid) + max_interactions_count_in_test]

    @staticmethod
    def _load_from_txt(path: Path | str, build_dict: bool = False):
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
    
    def prepare_train_data(self) -> None:
        data = uniform_sampler(data_size=self.data_size,
                               users_count=self.num_users,
                               user_pos_items_dict_train= self.user_pos_dict,
                               items_count=self.num_items,
                               neg_per_pos=self.neg_per_pos)
        self.data = torch.tensor(data, dtype=torch.long)

    def prepare_test_data(self, users_l, items_l) -> None:
        unique_users, inverse_idx = np.unique(users_l, return_inverse=True)
        counts = np.bincount(inverse_idx)
        max_len = counts.max() + 1
        arr = np.full((len(unique_users), max_len), -1, dtype=int)
        arr[:, 0] = unique_users

        offsets = np.zeros(len(unique_users), dtype=int)
        for idx, u in enumerate(inverse_idx):
            pos = offsets[u] + 1
            arr[u, pos] = items_l[idx]
            offsets[u] += 1
        self.data = torch.from_numpy(arr)

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X: csr_matrix) -> torch.sparse.FloatTensor:
        coo = X.tocoo().astype(np.float32)
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        index = torch.stack([row, col])
        data = torch.tensor(coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
    
    @staticmethod
    def _get_user_item_matrix(users: list[int], items: list[int],
                              n_row: int, n_col: int) -> csr_matrix:
        return csr_matrix((np.ones(len(users)), (users, items)),
                          shape=(n_row, n_col))
    
    def dupm_user_item_data(self,data) -> None:
        dataset_name = Path(self.data_path).parent.stem
        save_path = self.dump_dir / f"{dataset_name}.pt"
        torch.save(data, save_path)
        print(f"User-item interaction data dumped to {save_path}")
    
    def get_models_necessary_input(self) -> dict:
        raise NotImplementedError("Child classes must implement get_models_necessary_data()")
    

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]
    

class BipartiteGraphDataset(BaseUI_Interation_Dataset):
    def __init__(self,
                 data_path: Path | str,
                 is_train_data: bool = True,
                 neg_per_pos: int | None = None,
                 normalization_type: str = "symmetric",
                 add_self_loop: bool = False) -> None:

        if normalization_type not in ["symmetric", "asymmetric"]:
            raise ValueError(f"Invalid normalization_type: {normalization_type}")

        super().__init__(data_path=data_path,
                         is_train_data=is_train_data,
                         neg_per_pos=neg_per_pos)

        self.add_self_loop = add_self_loop
        self.normalization_type = normalization_type

    def _get_bipartite_adjacency(self, user_item_matrix_train: csr_matrix) -> csr_matrix:
        # A = [0, R;
        #      R^T, 0]
        adj_mat = sp.bmat([[None, user_item_matrix_train],
                           [user_item_matrix_train.transpose(), None]],
                          format='csr',
                          dtype=np.float32)

        if self.add_self_loop:
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0], dtype=np.float32)

        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)

        if self.normalization_type == "symmetric":
            return d_mat.dot(adj_mat).dot(d_mat)
        elif self.normalization_type == "asymmetric":
            return d_mat.dot(adj_mat)
        
    @override    
    def get_models_necessary_input(self,is_dump) -> torch.sparse.FloatTensor:
        user_item_matrix = self._get_user_item_matrix(self.users_l, self.items_l,
                                                      self.num_users, self.num_items)
        del self.users_l
        del self.items_l
        norm_adj = self._get_bipartite_adjacency(user_item_matrix)
        norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce()
        final_data = {"users_count": self.num_users,
                      "items_count": self.num_items,
                      "graph": norm_adj}

        if is_dump:
            save_path = self.dump_dir / f"BipartiteGraphDataset_model_input.pt"
            torch.save(final_data, save_path)
            print(f"Model input dumped to {save_path}\n")
            self.dupm_user_item_data(user_item_matrix)
        return final_data

    


class KnowledgeGraphDataset(BaseUI_Interation_Dataset):
    def __init__(self,
                 data_path: Path | str,
                 kg_path: Path | str| None = None,
                 neg_per_pos: int | None = None,
                 is_train_data: bool = True) -> None:

        super().__init__(data_path=data_path,
                         is_train_data=is_train_data,
                         neg_per_pos=neg_per_pos)
        
        if is_train_data:
            self.kg_path = kg_path
            self.triplets = self._read_triplets(kg_path)
            self.n_entities = max(max(self.triplets[:, 0]), max(self.triplets[:, 2])) + 1
            self.n_relations = max(self.triplets[:, 1])  # start index is 1

            print(f"Number of entities in KG: {self.n_entities}, Number of relations in KG: {self.n_relations}")
        else:
            pass

    @staticmethod
    def _read_triplets(file_name: str) -> np.ndarray:
        can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
        can_triplets_np = np.unique(can_triplets_np, axis=0)
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        return can_triplets_np

    @staticmethod
    def _get_row_normalized_matrix(user_item_matrix: csr_matrix) -> csr_matrix:
        rowsum = np.array(user_item_matrix.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        return d_mat_inv.dot(user_item_matrix)

    def get_interact_mat(self) -> torch.sparse.FloatTensor:
        # Override adjacency to use row-normalized UI matrix (KGIN-specific)
        user_item_matrix = self._get_user_item_matrix(self.users_l, self.items_l,
                                                      self.num_users, self.n_entities)
        del self.users_l
        del self.items_l
        norm_adj = self._get_row_normalized_matrix(user_item_matrix)
        return self._convert_sp_mat_to_sp_tensor(norm_adj).coalesce(), user_item_matrix
    

    def get_edge_index_and_type(self) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.tensor(self.triplets[:,[0,2]]).t().long()
        edge_type = torch.tensor(self.triplets[:,1]).long()
        del self.triplets
        return edge_index, edge_type
    
    @override
    def get_models_necessary_input(self,is_dump: bool = False):
        edge_index, edge_type = self.get_edge_index_and_type()
        norm_user_item_matrix ,user_item_matrix = self.get_interact_mat()
        final_data = {"users_count": self.num_users,
                      "items_count": self.n_entities,
                      "n_relations": self.n_relations,
                      "edge_index": edge_index,
                      "edge_type": edge_type,
                      "interact_mat": norm_user_item_matrix}
        if is_dump:
            save_path = self.dump_dir / f"KnowledgeGraphDataset_model_input.pt"
            torch.save(final_data, save_path)
            print(f"Model input dumped to {save_path}\n")
            self.dupm_user_item_data(user_item_matrix)

        return final_data
    