

import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F

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
                 user_pretrained_weights:bool = False,
                 pretrained_weights_path:Optional[str] = None):
        super().__init__()
        
        self.users_count = users_count
        self.items_count = items_count

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
        self.load_state_dict(state_dict)

    
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

        users = self.user_embedding(user_ids)
        pos_items = self.item_embedding(pos_item_ids)
        neg_items = self.item_embedding(neg_item_ids)

        reg_loss = (1/2)*(users.norm(2).pow(2) + 
                         pos_items.norm(2).pow(2)  +
                         neg_items.norm(2).pow(2))/ float(len(users))
        
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

        users_emb = all_users[user_ids] # 3 by 64
        pos_emb = all_items[pos_item_ids] # 3 by 64
        neg_emb = all_items[neg_item_ids] # 3 by 64

        pos_scores = torch.mul(users_emb, pos_emb) # 3 by 2
        pos_scores = torch.sum(pos_scores, dim=1)

        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

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
        return reg_loss, bpr_loss
    

    def forward(self,
                user_ids:int|list[int]) -> torch.Tensor:
        ''' either for single user [sim_1,sim_2. sim_3...sim_N]

            OR for multiple users [[sim_1_1,sim_2_1. sim_3_1...sim_N_1],
                                   [sim_1_2,sim_2_2. sim_3_2...sim_N_2],
                                   [...]

        '''
        
        all_users, all_items = self.propogate()
        users_emb = all_users[user_ids]
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
                 user_pretrained_weights: bool = False,
                 pretrained_weights_path: Optional[str] = None,
                 x_dropout_rate:float=0.2,
                 z_dropout_rate:float=0.2):
        
        super().__init__(users_count,
                         items_count,
                         graph,
                         device,
                         latent_dim,
                         n_layers,
                         user_pretrained_weights,
                         pretrained_weights_path)
        
        self.alpha = alpha
        self.beta = beta
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

            return reg_loss, loss