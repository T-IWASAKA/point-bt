# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 00:00:00 2025

ihvit module

@author: iwasakat
"""
import pickle
from typing import Tuple
import yaml
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
import torchvision.transforms as transforms
from transformers import get_linear_schedule_with_warmup

import wandb

#from .src.models.simsiam import SimSiam
#from .src.models.simslr import CimCLR_RBC
from .src.models.barlowtwins import BarlowTwins

from .src.data_handler import (
    get_clstoken,
    get_dr_feature,
    prep_smeardata_ssl,
    prep_smeardata_pointbt,
    prep_validdataset_bg_lst
)
from .src.image_aug import SSLTransform, SSLTransform2
from .src.models.vit import VitForPointBT
from .src.models.point_bt import PointBT
from .src.trainer import Trainer

class BTRBC:
    def __init__(
            self, config_path: str
            ):
        # configの読み込み
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.config["config_path"] = config_path
        self.input_path = None
        self.model = None
        self.backbone = None


    def load_model(self, model_path: str, config_path: str=None):
        """ モデルの読み込み """
        if config_path is not None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            self.config["config_path"] = config_path
        self.model = VitForPointBT(self.config)
        self.model.load_state_dict(torch.load(model_path))
    
    def prep_data(
            self, exp_name: str=None, input_path: str=None,
            num_rbc=2000, show_imagedata=True,
            num_workers=2, pin_memory=True,
            save_path="/"
            ):
        """ dataの読み込み, 背景画像でスライド間差を学習するためのデータセットを準備 """
        if exp_name is None:
            exp_name = "exp"
        self.config["exp_name"] = exp_name
        self.input_path = input_path
        ssltf = SSLTransform(crop_size=self.config["crop_size"])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        train_loader, test_loader = prep_smeardata_ssl(
            path=input_path, 
            num_rbc=num_rbc, 
            show_imagedata=show_imagedata,
            batch_size=self.config["batch_size"], 
            ssl_transform=ssltf,
            shuffle=(True, False), 
            num_workers=num_workers, 
            pin_memory=pin_memory, 
            save_path=save_path,
            )
        return train_loader, test_loader
    
    def prep_data_pointbt(
            self, exp_name: str=None, input_path: str=None,
            num_rbc=2000,
            num_workers=2, pin_memory=True,
            save_path="/"
            ):
        """ dataの読み込み, 背景画像でスライド間差を学習するためのデータセットを準備 """
        if exp_name is None:
            exp_name = "exp"
        self.config["exp_name"] = exp_name
        self.input_path = input_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        train_loader, test_loader = prep_smeardata_pointbt(
            path=input_path, 
            num_rbc=num_rbc, 
            batch_size=self.config["batch_size"], 
            shuffle=(True, False),
            num_workers=num_workers, 
            pin_memory=pin_memory, 
            save_path=save_path,
            ) 
        return train_loader, test_loader
    

    def prep_data_ds(
            self, image_path,
            num_image=2000, splitn=5,
            show_imagedata=True,
            dataset_save=None
            ):
        if os.path.exists(dataset_save+'/ds_dataset_bgcor_lst.pickle'): #
            with open(dataset_save+'/ds_dataset_bgcor_lst.pickle', 'rb') as f:
                dataset_lst = pickle.load(f)
        else:
            dataset_lst = prep_validdataset_bg_lst(image_path, num_image=num_image, splitn=splitn, show_imagedata=show_imagedata)
            with open(dataset_save+'/ds_dataset_bgcor_lst.pickle', 'wb') as f:
                pickle.dump(dataset_lst, f)

        return dataset_lst 


    def fit(self, train_loader, test_loader, btconfig={}, model="ss", warmup=True, run=None):
        """ training """
        # モデル等の準備
        self.latent_id = btconfig["latent_id"]
        if model == "ss":
            self.latent_id = btconfig["latent_id"]
            self.backbone = VitForPointBT(self.config)
            #self.model = SimSiam(self.backbone, self.latent_id, btconfig["projection_sizes"])
        elif model == "bgcor":
            self.latent_id = btconfig["latent_id"]
            self.backbone = VitForPointBT(self.config)
            #self.model = CimCLR_RBC(self.backbone, self.latent_id, btconfig["projection_sizes"], btconfig["lambd"], scale_factor=btconfig["scale_factor"])
        elif model == "bt":
            self.latent_id = btconfig["latent_id"]
            self.backbone = VitForPointBT(self.config)
            self.model = BarlowTwins(self.backbone, self.latent_id, btconfig["projection_sizes"], btconfig["lambd"], scale_factor=btconfig["scale_factor"])     
        elif model == "pointbt":
            self.latent_id = btconfig["latent_id"]
            self.model = PointBT(self.backbone, self.latent_id, self.backbone_projector, self.point_input_dim, btconfig["projection_sizes"], btconfig["lambd"], scale_factor=btconfig["scale_factor"])   
            #backbone_projector, point_input_dim については後からうまいこと定義する、一旦外から入れる
        else:
            self.model = VitForPointBT(self.config)

        # --- ▼ WandBの追記ここから ▼ ---
        # (オプション) モデルの勾配や構造を監視
        wandb.watch(self.model, log_freq=100)

        # CosineAnnealingLR + warmup の組み合わせ
        if warmup:
            # === ここから ===
            # Optimizer の定義
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-2)

            # ウォームアップとコサイン減衰を組み合わせる設定
            num_epochs = self.config["epochs"]
            num_training_steps = len(train_loader) * num_epochs
            num_warmup_steps = int(0.1 * num_training_steps)  # 例: 全体の10%をウォームアップにする

            # ウォームアップスケジューラー
            warmup_scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )

            # コサイン減衰スケジューラー (ウォームアップ後に使う)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - (num_warmup_steps / len(train_loader)))

            # スケジューラーを連結する
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[num_warmup_steps]
            )
            # === ここまで ===

        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(
            self.config, self.model, optimizer, scheduler, loss_fn, self.config["exp_name"], self.config["device"], run
            )
        # training
        trainer.train(
            train_loader, test_loader, save_model_evry_n_epochs=self.config["save_model_every"]
            )
        
        avg_loss, avg_on_diag, avg_off_diag = trainer.evaluate(test_loader)
        print(f"Average Loss: {avg_loss}")
        print(f"Best epoch: {trainer.best_epoch}")
        self.best_epoch = trainer.best_epoch

        # --- ▼ WandBの追記ここから ▼ ---
        # 2. Artifactを作成
        base_dir = "/workspace/wandbdata" # trainerの部分もハードになっている、後で変更したい
        artifact = wandb.Artifact(
            name=self.config["exp_name"], # 'my-vit-model'のような管理しやすい名前
            type="model",
            metadata={"best_epoch": self.best_epoch, "config": self.config, "btconfig": btconfig} # メタデータも記録可能
            )
        best_model_path = f"{base_dir}/{self.config['exp_name']}/model_{self.best_epoch}.pt"
        artifact.add_file(best_model_path)
        run.log_artifact(artifact)
        # --- ▲ WandBの追記ここまで ▲ ---

        best_weights = torch.load(best_model_path)
        self.backbone.load_state_dict(best_weights)

        return self.backbone


    def model_valid(self, dataset_lst, latent_id, f_type="mean+max"):
        sorted_features = get_clstoken(dataset_lst, self.model, self.config["device"], f_type=f_type, latent_id=latent_id, batch_size=64)
        pca_feature, sample_ind = get_dr_feature(sorted_features)
        return  pca_feature, sample_ind
    

