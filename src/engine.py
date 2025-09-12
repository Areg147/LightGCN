
import torch
import faiss
import numpy as np
import warnings
from typing import Union
from config import LightGCNConfig, MGDCFConfig
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class InferenceEngine:
    """A wrapper for fast inference, with FAISS search and bulk score precomputation."""
    def __init__(self, config: Union[LightGCNConfig, MGDCFConfig]):
        self.config = config
        self.device = self.config.device
        self.model = self.config.build_model().to(self.device)
        
        if self.config.use_pretrained_weights and self.config.pretrained_weights_path:
            try:
                print(f"  > Loading pretrained weights from: {self.config.pretrained_weights_path}")
                self.model._load_pretrained_weights()
            except FileNotFoundError:
                raise IOError(f"Pretrained weights file not found: {self.config.pretrained_weights_path}")
        else:
            warnings.warn("No pretrained weights provided. Model is using random weights.", UserWarning)
            
        self.model.eval()
        self._cache_embeddings_and_build_faiss()
        print("--- Engine Initialized Successfully ---")

    @torch.no_grad()
    def _cache_embeddings_and_build_faiss(self):
        print("  > Caching final embeddings and building FAISS index...")

        self.user_embeddings, self.item_embeddings = self.model.propogate()
        item_embeddings_np = self.item_embeddings.cpu().numpy().astype('float32')
        self.faiss_index = faiss.IndexFlatL2(item_embeddings_np.shape[1])
        self.faiss_index.add(item_embeddings_np)

        print(f"  > FAISS index built. Total items: {self.faiss_index.ntotal}")

    @torch.no_grad()
    def recommend(self, user_ids: Union[int, list[int]], k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(user_ids, int): user_ids = [user_ids]
        query_vectors = self.user_embeddings[user_ids].cpu().numpy().astype('float32')
        return self.faiss_index.search(query_vectors, k)


    @torch.no_grad()
    def precompute_and_save_all_scores(self, output_path: str):
        print(f"\n--- Starting Precomputation of All Scores ---")
        
        num_users, num_items = self.config.users_count, self.config.items_count
        print(f"  > INFO: This will create a matrix of shape ({num_users}, {num_items}).")
        
        all_user_ids = torch.arange(num_users, device=self.device)
        all_scores = self.model(all_user_ids)

        scores_numpy = all_scores.cpu().numpy().astype('float32')
        np.save(output_path, scores_numpy)

        print(f"✅ Success! Full score matrix saved to '{output_path}'.")