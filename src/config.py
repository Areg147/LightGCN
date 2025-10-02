import sys
sys.path.append("./")
import yaml
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union
from src.models import LightGCN, MGDCF, KGIN  
@dataclass(frozen=True)
class BaseGraphModelsConfig:
    # --- Runtime Parameters ---
    users_count: int
    items_count: int
    device: str
    
    # --- File-based Hyperparameters ---
    model_name: str
    latent_dim: int
    n_layers: int
    reg_decay: float
    
    def build_model(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement the build_model method.")


@dataclass(frozen=True)
class LightGCNConfig(BaseGraphModelsConfig):
    graph: torch.Tensor
    use_pretrained_weights: bool = False
    pretrained_weights_path: Optional[str] = None
    
    
    def build_model(self) -> "LightGCN":
        return LightGCN(
            users_count=self.users_count,
            items_count=self.items_count,
            reg_decay=self.reg_decay,
            graph=self.graph,
            device=self.device,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            user_pretrained_weights=self.use_pretrained_weights,
            pretrained_weights_path=self.pretrained_weights_path
        )


@dataclass(frozen=True)
class MGDCFConfig(BaseGraphModelsConfig):
    # --- File-based Hyperparameters ---
    alpha: float
    beta: float
    x_dropout_rate: float
    z_dropout_rate: float
    graph: torch.Tensor
    use_pretrained_weights: bool = False
    pretrained_weights_path: Optional[str] = None
    

    def build_model(self) -> "MGDCF":
        return MGDCF(
            users_count=self.users_count,
            items_count=self.items_count,
            graph=self.graph,
            device=self.device,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            user_pretrained_weights=self.use_pretrained_weights,
            pretrained_weights_path=self.pretrained_weights_path,
            alpha=self.alpha,
            beta=self.beta,
            x_dropout_rate=self.x_dropout_rate,
            z_dropout_rate=self.z_dropout_rate,
            reg_decay=self.reg_decay
        )

@dataclass(frozen=True)
class KGINConfig(BaseGraphModelsConfig):
    # --- File-based Hyperparameters ---
    reg_decay: float
    sim_decay: float
    n_factors: int
    node_dropout_rate: float
    mess_dropout_rate: float
    ind: str  # "distance" or "cosine"

    # --- Runtime Parameters ---
    n_relations: int 
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    interact_mat: torch.Tensor
    use_pretrained_weights: bool = False
    pretrained_weights_path: Optional[str] = None
    def build_model(self) -> "KGIN":
        return KGIN(
            n_users=self.users_count,
            n_items=self.items_count, # n_items + n_items
            n_relations=self.n_relations,   
            sim_decay=self.sim_decay,
            reg_decay = self.reg_decay,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            n_factors=self.n_factors,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate,
            ind=self.ind,
            edge_index=self.edge_index,       
            edge_type=self.edge_type,        
            device=self.device,
            interact_mat=self.interact_mat,
            user_pretrained_weights=self.use_pretrained_weights,
            pretrained_weights_path=self.pretrained_weights_path
        )


def load_config(
    yaml_path: str,
    **runtime_params
) -> Union[LightGCNConfig, MGDCFConfig, KGINConfig]:
    with open(yaml_path, 'r') as file:
        yaml_params = yaml.safe_load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Automatically selected device: {device.upper()}")

    runtime_params["device"] = device
    all_params = {**yaml_params, **runtime_params}
    
    model_name = all_params.get("model_name")
    
    if model_name == "LightGCN":
        return LightGCNConfig(**all_params)
    elif model_name == "MGDCF":
        return MGDCFConfig(**all_params)
    elif model_name == "KGIN":
        return KGINConfig(**all_params)
    else:
        raise ValueError(f"Unknown model_name '{model_name}' in {yaml_path}")