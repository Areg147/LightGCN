

import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from typing import Optional
from torch.nn import functional as F
import numpy as np


def strip_first_prefix_inplace(state_dict) -> None:
    for k in list(state_dict.keys()):  # list() because we'll change keys
        if '.' in k:
            new_key = k.split('.', 1)[1]
        else:
            new_key = k
        if new_key != k:
            state_dict[new_key] = state_dict.pop(k)


class LightGCN(nn.Module):
    """ Class Wraps the LightGCN Model https://arxiv.org/pdf/2002.02126
        itinializes the user and item embeddings or loads pretrained
        
        1.Take input graph(Usert-Item Interaction Matrix)
        2.Propogate the embeddings through the graph
        3.Compute the Total Loss (BPR Loss + Regularization Loss)
    """
    def __init__(self,
                 users_count:int,
                 items_count:int,
                 graph:torch.Tensor,
                 device:str,
                 latent_dim:int,
                 n_layers:int,
                 reg_decay:float,
                 user_pretrained_weights:bool = False,
                 pretrained_weights_path:Optional[str] = None):
        super().__init__()
        
        self.users_count = users_count
        self.items_count = items_count
        self.reg_decay = reg_decay

        self.Graph = graph.to(device)

        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.user_pretrained_weights = user_pretrained_weights
        self.pretrained_weights_path = pretrained_weights_path

        self.user_embedding = nn.Embedding(self.users_count,
                                           self.latent_dim)
        self.item_embedding = nn.Embedding(self.items_count,
                                           self.latent_dim)
        self.f = nn.Sigmoid()

        if user_pretrained_weights:
            self._load_pretrained_weights()
        



    def _load_pretrained_weights(self):
        state_dict = torch.load(self.pretrained_weights_path)
        strip_first_prefix_inplace(state_dict["state_dict"])
        self.load_state_dict(state_dict["state_dict"])

    
    def propogate(self):

        ''' self.dl = data_loadergraph = (D^(-1/2) * A * D^(-1/2))
            E^(k+1) = (D^(-1/2) * A * D^(-1/2)) * E^(k)
            E^(0) = [users_emb]
                    [items_emb]

            E = α₀ E⁽⁰⁾ + α₁ E⁽¹⁾ + α₂ E⁽²⁾ + ... + α_K E⁽ᴷ⁾
            here is α's are  not learnable parameters, they are all 1/K 
        '''

        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = []
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb) # E⁽⁰⁾, E⁽¹⁾,E⁽²⁾..E⁽n_layer⁾
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1) 
        users, items = torch.split(light_out, [self.users_count,
                                               self.items_count])
        return users, items
    

    def _compute_reg_loss(self,
                         user_ids,
                         pos_item_ids,
                         neg_item_ids):
        
        batch_size = user_ids.shape[0]
        users = self.user_embedding(user_ids)
        pos_items = self.item_embedding(pos_item_ids)
        neg_items = self.item_embedding(neg_item_ids)

        reg_loss = (1/2)*(users.norm(2).pow(2) + 
                         pos_items.norm(2).pow(2)  +
                         neg_items.norm(2).pow(2))/ batch_size
        
        return reg_loss
    

    def _compute_bpr_loss(self,
                         user_ids,
                         pos_item_ids,
                         neg_item_ids):
        ''' NOTE input user_ids is a list of user_ids and for 
            each user_id there is a pos_item_id and 
            neg_item_id will be used to compute the BPR Loss
            ex [user_1, user_2, user_2]
               [pos_item_1, pos_item_2_1, pos_item_2_2]
               [neg_item_1, neg_item_2_1, neg_item_2_2]
        '''
        all_users, all_items = self.propogate()

        users_emb = all_users[user_ids] 
        pos_emb = all_items[pos_item_ids] 
        neg_emb = all_items[neg_item_ids]

        # Handle single negative vs multiple negatives
        if neg_emb.dim() == 2:  
            # [N, D] → add K=1 dim
            neg_emb = neg_emb.unsqueeze(1)  # [N, 1, D]

        users_expanded = users_emb.unsqueeze(1).expand_as(neg_emb)  # [N, K, D] 
        pos_scores = torch.mul(users_emb, pos_emb) # 3 by 2
        pos_scores = torch.sum(pos_scores, dim=1,keepdim=True)
        neg_scores = torch.mul(users_expanded, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=2)
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return bpr_loss
    

    def compute_loss(self,
                     user_ids,
                     pos_item_ids,
                     neg_item_ids):
        
        # Compute Regularization Loss on Embeddings
        reg_loss = self._compute_reg_loss(user_ids,
                                          pos_item_ids,
                                          neg_item_ids)
        
        # Compute BPR Loss on propogated Embeddings
        bpr_loss = self._compute_bpr_loss(user_ids,
                                          pos_item_ids,
                                          neg_item_ids)
        return self.reg_decay * reg_loss + bpr_loss
    

    def forward(self,
                user_ids:int|list[int]) -> torch.Tensor:
        ''' either for single user [sim_1,sim_2. sim_3...sim_N]

            OR for multiple users [[sim_1_1,sim_2_1. sim_3_1...sim_N_1],
                                   [sim_1_2,sim_2_2. sim_3_2...sim_N_2],
                                   [...]

        '''

        all_users, all_items = self.propogate()
        users_emb = all_users[user_ids]
        a = torch.matmul(users_emb, all_items.t())
        a = torch.sort(a, descending=True)
        rating = self.f(torch.matmul(users_emb, all_items.t()))
        return rating

    
class MGDCF(LightGCN):
    """ Inherits from LightGCN, applies MGDCF message passing """
    def __init__(self,
                 users_count: int,
                 items_count: int,
                 graph: torch.Tensor,
                 device: str,
                 latent_dim: int,
                 n_layers: int,
                 alpha: float,
                 beta: float,
                 reg_decay: float ,
                 user_pretrained_weights: bool = False,
                 pretrained_weights_path: Optional[str] = None,
                 x_dropout_rate:float=0.2,
                 z_dropout_rate:float=0.2):
        super().__init__(
                 reg_decay=reg_decay,
                 users_count=users_count,
                 items_count=items_count,
                 graph=graph,
                 device=device,
                 latent_dim=latent_dim,
                 n_layers=n_layers,
                 user_pretrained_weights=user_pretrained_weights,
                 pretrained_weights_path=pretrained_weights_path)  
        self.alpha = alpha
        self.beta = beta
        self.reg_decay = reg_decay
        self.gamma = beta**n_layers + alpha * sum(beta**i for i in range(n_layers))
        self.x_dropout = torch.nn.Dropout(p=x_dropout_rate)
        self.z_dropout = torch.nn.Dropout(p=z_dropout_rate)

    def propogate(self):
        """
        MGDCF-style propagation:
        h_k = beta * A h_{k-1} + alpha * h_0
        Final output: h_k / gamma
        """
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        h0 = self.x_dropout(torch.cat([users_emb, items_emb], dim=0))
        h = h0

        for _ in range(self.n_layers):
            Ah = torch.sparse.mm(self.Graph, h)
            h = self.beta * Ah + self.alpha * h0

        h = h / self.gamma
        h = self.z_dropout(h)
        users, items = torch.split(h, [self.users_count, self.items_count], dim=0)
        return users, items

    def compute_loss(self,
                    user_ids,
                    pos_item_ids,
                    neg_item_ids):
            """
            user_ids:       Tensor [N]
            pos_item_ids:   Tensor [N]
            neg_item_ids:   Tensor [N, K]
            """
            all_users, all_items = self.propogate()  # get full user/item embeddings

            users_emb = all_users[user_ids]                  # [N, D]
            pos_emb = all_items[pos_item_ids]                # [N, D]
            neg_emb = all_items[neg_item_ids]                # [N, K, D]

            # Expand user embeddings to [N, K, D]
            users_expanded = users_emb.unsqueeze(1).expand_as(neg_emb)  # [N, K, D]

            # -------- POSITIVE SCORES --------
            pos_scores = torch.sum(users_emb * pos_emb, dim=1, keepdim=True)  # [N, 1]

            # -------- NEGATIVE SCORES --------
            neg_scores = torch.sum(users_expanded * neg_emb, dim=2)  # [N, K]

            # -------- CONCAT + CROSS ENTROPY --------
            logits = torch.cat([pos_scores, neg_scores], dim=1)  # [N, 1 + K]
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positives at index 0
            loss = F.cross_entropy(logits, targets, reduction='mean')

            user_norm = users_emb.norm(p=2, dim=1).pow(2).sum()          # scalar
            pos_item_norm = pos_emb.norm(p=2, dim=1).pow(2).sum()  # scalar
            neg_item_norm = neg_emb.norm(p=2, dim=2).pow(2).sum()  # [N], then sum over all
            reg_loss = 0.5 * (user_norm + pos_item_norm + neg_item_norm.sum()) / float(user_ids.size(0))

            return self.reg_decay*reg_loss + loss
    
class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self,
                 n_users,
                 n_factors,
                 n_items,
                 latent_dim):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors
        self.n_items = n_items
        self.latent_dim = latent_dim
        self.f_1 = nn.Softmax(dim=1)
        self.f_2 = nn.Softmax(dim=-1)


    def forward(self,
                entity_emb, 
                user_emb,
                latent_emb, # of shape 1 by latent_dim (requires_grad=True)
                edge_index, # items:list[int],  entitys:list[int]
                edge_type, # items <--> entity connection type
                interact_mat, # user-item normalized interact_mat
                weight, # of shape np.nuique(edge_type) by latent_dim (requires_grad=True)
                disen_weight_att # of shape 1 by np.nuique(edge_type) (requires_grad=True)
                ):

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  

        entity_agg = scatter_mean(src=neigh_relation_emb,
                                  index=head,
                                  dim_size=self.n_items,
                                  dim=0)

        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_emb.t())
        score = self.f_1(score_).unsqueeze(-1)  

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)

        disen_weight_1 = self.f_2(disen_weight_att)  
        disen_weight_2 = torch.mm(disen_weight_1, weight)

        disen_weight_expanded = disen_weight_2.expand(self.n_users,
                                                      self.n_factors,
                                                      self.latent_dim)
        
        user_agg = user_agg * (disen_weight_expanded * score).sum(dim=1) + user_agg  
        return entity_agg, user_agg
    
class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self,
                 latent_dim:int,
                 n_layers:int,
                 n_users:int,
                 n_factors:int,
                 n_relations:int,
                 n_items:int,
                 node_dropout_rate:float=0.5,
                 mess_dropout_rate:float=0.1,
                 ind:str="distance"
                 ):
        
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations, latent_dim))  
        self.weight = nn.Parameter(weight)  

        disen_weight_att = initializer(torch.empty(n_factors, n_relations))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_layers):
            self.convs.append(Aggregator(n_users=n_users,
                                         n_factors=n_factors,
                                         n_items=n_items,
                                         latent_dim=latent_dim))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  #message dropout


    @staticmethod
    def _edge_sampling(edge_index:torch.tensor, # 2 by n_items
                       edge_type:torch.tensor, # 1 by n_items
                       rate:float):
        
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]
    

    @staticmethod
    def _sparse_dropout(x:torch.sparse_coo_tensor, # user-item normalized interaction matrix
                        rate:float):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse_coo_tensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))
    
    @staticmethod
    def _cosinesimilarity(tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2

    @staticmethod
    def _distance_correlation(tensor_1: torch.Tensor,
                             tensor_2: torch.Tensor) -> torch.Tensor:
        """
        Compute the distance correlation between two tensors.
        Reference: https://en.wikipedia.org/wiki/Distance_correlation

        Args:
            tensor_1 (Tensor): shape [N], first 1D vector.
            tensor_2 (Tensor): shape [N], second 1D vector.

        Returns:
            Tensor: scalar tensor with the distance correlation value.
        """
        n = tensor_1.shape[0]
        device = tensor_1.device

        zeros_matrix = torch.zeros(n, n, device=device)
        zero_scalar = torch.zeros(1, device=device)

        # Expand to [N, 1] for pairwise operations
        tensor_1 = tensor_1.unsqueeze(-1)
        tensor_2 = tensor_2.unsqueeze(-1)

        # ---- Step 1: Compute pairwise distance matrices ----
        # Using (x_i - x_j)^2 = x_i^2 + x_j^2 - 2 x_i x_j
        dot_x = 2 * torch.matmul(tensor_1, tensor_1.t())
        dot_y = 2 * torch.matmul(tensor_2, tensor_2.t())

        sq_x = tensor_1 ** 2
        sq_y = tensor_2 ** 2

        dist_x = torch.sqrt(
            torch.max(sq_x - dot_x + sq_x.t(), zeros_matrix) + 1e-8
        )
        dist_y = torch.sqrt(
            torch.max(sq_y - dot_y + sq_y.t(), zeros_matrix) + 1e-8
        )

        # ---- Step 2: Double-center the distance matrices ----
        A = dist_x - dist_x.mean(dim=0, keepdim=True) \
                    - dist_x.mean(dim=1, keepdim=True) \
                    + dist_x.mean()
        B = dist_y - dist_y.mean(dim=0, keepdim=True) \
                    - dist_y.mean(dim=1, keepdim=True) \
                    + dist_y.mean()

        # ---- Step 3: Compute distance covariance terms ----
        dcov_ab = torch.sqrt(
            torch.max((A * B).sum() / (n ** 2), zero_scalar) + 1e-8
        )
        dcov_aa = torch.sqrt(
            torch.max((A * A).sum() / (n ** 2), zero_scalar) + 1e-8
        )
        dcov_bb = torch.sqrt(
            torch.max((B * B).sum() / (n ** 2), zero_scalar) + 1e-8
        )

        # ---- Step 4: Normalize to get distance correlation ----
        dcor = dcov_ab / torch.sqrt(dcov_aa * dcov_bb + 1e-8)

        return dcor
    
    def claculate_cor(self):
        cor = 0
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                if self.ind == 'distance':
                    cor += self._distance_correlation(self.disen_weight_att[i],
                                                      self.disen_weight_att[j])
                else:
                    cor += self._cosinesimilarity(self.disen_weight_att[i],
                                                  self.disen_weight_att[j])
        return cor

    def forward(self,
                user_emb,
                entity_emb,
                latent_emb,
                edge_index,
                edge_type,
                interact_mat):

        """node dropout"""
        if self.node_dropout_rate>0:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, latent_dim]
        user_res_emb = user_emb  # [n_users, latent_dim]
        correlation_score = self.claculate_cor()
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, self.disen_weight_att)
            if self.dropout.p > 0:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, correlation_score
    
class KGIN(nn.Module):
    def __init__(self,
                 n_users: int,
                 n_items: int, # n_items + n_entities
                 n_relations: int,
                 reg_decay: float,
                 sim_decay: float,
                 latent_dim: int,
                 n_layers: int,
                 n_factors: int,
                 node_dropout_rate: float,
                 mess_dropout_rate: float,
                 ind: str,
                 edge_index,
                 edge_type,
                 device,
                 interact_mat,
                 user_pretrained_weights:bool = False,
                 pretrained_weights_path:Optional[str] = None):
        super(KGIN, self).__init__()


        self.user_pretrained_weights = user_pretrained_weights
        self.pretrained_weights_path = pretrained_weights_path

        # Dataset parameters
        self.n_items = n_items
        self.n_relations = n_relations

        # Regularization / model hyperparams
        self.reg_decay = reg_decay # for L2 reg
        self.sim_decay = sim_decay # for correlation 
        self.n_layers = n_layers


        self.device = device
        # data

        self.interact_mat = interact_mat.to(self.device)
        self.edge_index = edge_index
        self.edge_type = edge_type


        self.user_emb = nn.Embedding(n_users,
                                     latent_dim)
        self.item_emb = nn.Embedding(n_items,
                                     latent_dim)
        
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(1, latent_dim))  
        self.latent_emb = nn.Parameter(weight)
        self.f = nn.Sigmoid() 


        self.gcn = GraphConv(latent_dim=latent_dim,
                             n_items = n_items,
                             n_layers=n_layers,
                             n_users=n_users,
                             n_relations=n_relations,
                             n_factors=n_factors,
                             ind=ind)
        

        if user_pretrained_weights:
            self._load_pretrained_weights()
        
    def _load_pretrained_weights(self):
        state_dict = torch.load(self.pretrained_weights_path)
        strip_first_prefix_inplace(state_dict["state_dict"])
        self.load_state_dict(state_dict["state_dict"])


    def propogate(self):
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(self.user_emb.weight,
                                                     self.item_emb.weight,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat)
        return user_gcn_emb, entity_gcn_emb, cor


    def compute_loss(self, user, pos_item, neg_item):
        entity_gcn_emb, user_gcn_emb, cor = self.propogate()
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        neg_e.squeeze_(dim=1)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        u_e_expanded = u_e.unsqueeze(1)
        neg_scores = torch.sum(torch.mul(u_e_expanded, neg_e), axis=2)
        # mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores.unsqueeze(1) - neg_scores))
        mf_loss = torch.nn.functional.softplus(neg_scores - pos_scores.unsqueeze(1))
        mf_loss = torch.mean(mf_loss)

        # cul regularizer
        regularizer = (torch.norm(u_e) ** 2
                       + torch.norm(pos_e) ** 2
                       + torch.norm(neg_e) ** 2) / 2
        emb_loss = self.reg_decay * regularizer / len(user)
        cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss + cor_loss
    
    def forward(self,
                user_ids:int|list[int]) -> torch.Tensor:

        all_items, all_users ,_ = self.propogate()
        users_emb = all_users[user_ids]
        a = torch.matmul(users_emb, all_items.t())
        a = torch.sort(a, descending=True)
        rating = self.f(torch.matmul(users_emb, all_items.t()))
        return rating