import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
import pytorch_lightning as pl
from collections import defaultdict
from torch.utils.data import DataLoader
from typing import Callable


OPTIMIZER_REGISTRY = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSprop,
}

SCHEDULER_REGISTRY = {
    "cosine": CosineAnnealingLR,
    "step": StepLR,
    "onecycle": OneCycleLR,
    "reduce_on_plateau": ReduceLROnPlateau,
}


class RecSysTrainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        metric_calculator: Callable,
        train_dataset,
        val_dataset,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.metric_calculator = metric_calculator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.validation_metrics = defaultdict(list)
        self.save_hyperparameters(config)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        user_ids, pos_item_ids, neg_item_ids = batch[:, 0], batch[:, 1], batch[:, 2:]
        reg_loss, bpr_loss = self.model.compute_loss(
            user_ids, pos_item_ids, neg_item_ids.squeeze(-1)
        )
        loss = bpr_loss + self.config["reg_loss_weight"] * reg_loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=False,
            batch_size=batch.size(0),
        )
        grad_norm = torch.nn.utils.get_total_norm(self.model.parameters())
        self.log("grad_norm", grad_norm, on_step=True, prog_bar=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        user_ids, pos_item_ids = batch[:, 0], batch[:, 1:]
        ratings = self.model(user_ids)
        preds = torch.argsort(ratings, descending=True)
        metrics = self.metric_calculator(
            user_preds=preds,
            user_gts=pos_item_ids,
            top_ks=self.config["top_k"],
        )
        for metric_name, value in metrics.items():
            self.validation_metrics[metric_name].append(value)

    def on_validation_epoch_end(self) -> None:
        for metric_name, values in self.validation_metrics.items():
            avg_value = sum(values) / len(values)
            self.log(f"val_{metric_name}", avg_value, prog_bar=True)
        self.validation_metrics.clear()

    def configure_optimizers(self):
        opt_name = self.config.get("optimizer", "adam").lower()
        optimizer_cls = OPTIMIZER_REGISTRY[opt_name]
        optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get("weight_decay", 0.0),
        )
        scheduler_cfg = self.config.get("scheduler", None)
        if scheduler_cfg:
            sch_name = scheduler_cfg["name"].lower()
            scheduler_cls = SCHEDULER_REGISTRY[sch_name]
            if sch_name == "cosine":
                scheduler = scheduler_cls(
                    optimizer, T_max=scheduler_cfg.get("t_max", 10)
                )
            elif sch_name == "step":
                scheduler = scheduler_cls(
                    optimizer,
                    step_size=scheduler_cfg.get("step_size", 10),
                    gamma=scheduler_cfg.get("gamma", 0.1),
                )
            elif sch_name == "onecycle":
                scheduler = scheduler_cls(
                    optimizer,
                    max_lr=scheduler_cfg.get("max_lr", 0.01),
                    steps_per_epoch=len(self.train_dataloader()),
                    epochs=self.config.get("max_epochs", 10),
                )
            elif sch_name == "reduce_on_plateau":
                scheduler = scheduler_cls(
                    optimizer,
                    mode="max",
                    factor=scheduler_cfg.get("factor", 0.1),
                    patience=scheduler_cfg.get("patience", 5),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": scheduler_cfg.get("monitor", "val_loss"),
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            return [optimizer], [scheduler]
        return optimizer

    def train_dataloader(self) -> DataLoader:
        self.train_dataset.prepare_train_data()
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )