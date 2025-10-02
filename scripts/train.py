import sys
sys.path.append("./")
import os
import yaml
import mlflow
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchviz import make_dot
from src.data_loader import BipartiteGraphDataset,KnowledgeGraphDataset
from src.trainer import RecSysTrainer
from src.config import load_config
from src.utils import ndcg_recall_precision_batch
import argparse


def _load_from_yaml(path: str) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_trainer_and_config(
    model_config_path: str,
    model_name: str,
    trainer_config_path: str,
    train_data_path: str,
    val_data_path: str,
) -> tuple[RecSysTrainer, dict]:
    
    trainer_config = _load_from_yaml(trainer_config_path)
    neg_per_pos = trainer_config.get("neg_per_pos")
    is_add_self_loop = trainer_config.get("add_self_loop")
    normalization_type = trainer_config.get("normalization_type")
    if model_name in ["lightgcn", "mgdcf"]: 
        train_dataset = BipartiteGraphDataset(
            data_path=train_data_path,
            is_train_data=True,
            add_self_loop=is_add_self_loop,
            neg_per_pos=neg_per_pos,
            normalization_type = normalization_type
        )
        val_dataset = BipartiteGraphDataset(
            data_path=val_data_path,
            is_train_data=False,
            add_self_loop=is_add_self_loop,
        )
    elif model_name == "kgin":
        kg_path = f"data/{args.dataset_name}/kg_final.txt"
        train_dataset = KnowledgeGraphDataset(
            kg_path = kg_path,
            data_path=train_data_path,
            is_train_data=True,
            neg_per_pos=neg_per_pos,
        )
        val_dataset = KnowledgeGraphDataset(
            data_path=val_data_path,
            is_train_data=False,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    is_dump = trainer_config["dump_model_inputs_data"]
    model_inputs = train_dataset.get_models_necessary_input(is_dump=is_dump)
    model = load_config(
        yaml_path=model_config_path,
        **model_inputs).build_model()


    metric_calculator = ndcg_recall_precision_batch
    trainer = RecSysTrainer(
        model=model,
        config=trainer_config,
        metric_calculator=metric_calculator,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    return trainer, trainer_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (folder under data/)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (used for config path)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config_path = f"src/models_configs/{args.model_name}_config.yaml"
    trainer_config_path = "scripts/trainer_config.yaml" 
    train_data_path = f"data/{args.dataset_name}/train.txt"
    val_data_path = f"data/{args.dataset_name}/val.txt"

    trainer, train_configs = get_trainer_and_config(
        model_config_path=model_config_path,
        trainer_config_path=trainer_config_path,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        model_name=args.model_name
    )

    tracking_uri = "train_logs/logs"
    mlflow.set_tracking_uri(f"file:{tracking_uri}")

    mlflow_logger = MLFlowLogger(
        experiment_name=train_configs["experiment_name"],
        run_name=train_configs["run_name"],
        tracking_uri=f"file:{tracking_uri}",
    )

    experiment_name = train_configs["experiment_name"]
    run_name = train_configs["run_name"]
    folder_name = f"{experiment_name}__{run_name}"
    checkpoint_dir = os.path.join("train_logs/checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    callback = ModelCheckpoint(
        monitor=train_configs["monitor_metric"],
        mode="max",
        save_top_k=train_configs["save_top_k"],
        dirpath=checkpoint_dir,
        auto_insert_metric_name=True,
        save_weights_only=train_configs.get("save_weights_only"),
    )

    trainer_pl = pl.Trainer(
        logger=mlflow_logger,
        callbacks=[callback],
        max_epochs=train_configs["max_epochs"],
        log_every_n_steps=train_configs["log_every_n_steps"],
        accelerator=device,
        check_val_every_n_epoch=train_configs["check_val_every_n_epoch"],
        reload_dataloaders_every_n_epochs=train_configs["reload_dataloaders_every_n_epochs"]
    )
    trainer_pl.fit(trainer)

    mlflow.end_run()