import yaml
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union
from models import LightGCN, MGDCF

@dataclass(frozen=True)
class BaseConfig:
    # --- Runtime Parameters ---
    users_count: int
    items_count: int
    graph: torch.Tensor
    device: str
    
    # --- File-based Hyperparameters ---
    model_name: str
    latent_dim: int
    n_layers: int
    
    def build_model(self) -> nn.Module:
        raise NotImplementedError("Subclasses must implement the build_model method.")

@dataclass(frozen=True)
class LightGCNConfig(BaseConfig):
    use_pretrained_weights: bool = False
    pretrained_weights_path: Optional[str] = None
    
    def build_model(self) -> "LightGCN":
        return LightGCN(
            users_count=self.users_count,
            items_count=self.items_count,
            graph=self.graph,
            device=self.device,
            latent_dim=self.latent_dim,
            n_layers=self.n_layers,
            user_pretrained_weights=self.use_pretrained_weights,
            pretrained_weights_path=self.pretrained_weights_path
        )

@dataclass(frozen=True)
class MGDCFConfig(BaseConfig):
    # --- File-based Hyperparameters ---
    alpha: float
    beta: float
    x_dropout_rate: float
    z_dropout_rate: float
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
            z_dropout_rate=self.z_dropout_rate
        )

def load_config(
    yaml_path: str,
    users_count: int,
    items_count: int,
    graph: torch.Tensor,
) -> Union[LightGCNConfig, MGDCFConfig]:
    """
    Loads hyperparameters from YAML and combines them with runtime parameters.
    """
    with open(yaml_path, 'r') as file:
        yaml_params = yaml.safe_load(file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Automatically selected device: {device.upper()}")

    runtime_params = {
        "users_count": users_count,
        "items_count": items_count,
        "graph": graph,
        "device": device
    }
    all_params = {**yaml_params, **runtime_params}
    
    model_name = all_params.get("model_name")
    
    if model_name == "LightGCN":
        return LightGCNConfig(**all_params)
    elif model_name == "MGDCF":
        return MGDCFConfig(**all_params)
    else:
        raise ValueError(f"Unknown model_name '{model_name}' in {yaml_path}")

