# -*- coding: utf-8 -*-
"""
Created 2025

data handler

@author: iwasaka14
"""

## import ##
import random
import pickle
import itertools
from collections import deque
from typing import Tuple
import os
import sys
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
import torch
import torch.utils.data as data
from torch.utils.data import random_split
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

from .RBC_loader import Smear_tiff, show_get_img, show_get_img2


class SmearDataset_RBC_BG(torch.utils.data.Dataset):
    """
    # 赤血球とその近辺の背景画像をまとめたデータセット

    """
    def __init__(
            self,
            image_lst
        ):
        self.image_lst = image_lst
        
    def __len__(self):
        return len(self.image_lst)
    
    def __getitem__(self, idx):
        image_rbc = self.image_lst[idx][0]
        image_bg = self.image_lst[idx][1]

        return image_rbc, image_bg # ラベルとして背景画像を返す


class Dataset_SSL_add_BG(torch.utils.data.Dataset): # 二つの画像をセットで同じaugをかける
    """
    # SSL用にAugをかけた二つの画像をセットにし、まとめたデータセット (背景もあるため2x2=4枚で1セット)
    # BT用、Aug用のtransformはimage_aug.pyに記載
    
    """
    def __init__(self, mydataset, transform):
        if transform is None:
            raise ValueError('!! Give transform !!')
        self.transform = transform
        if len(mydataset) > 1:
            raise ValueError('!! Add a dataset that you have not SPLIT! !!')
        self.input = mydataset[0] #mydatasetがlist形式のためmydataset[0]であることに注意　


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        rbc_np, bg_np = self.input[idx]
        rbc = Image.fromarray(rbc_np) 
        bg = Image.fromarray(bg_np)
        y1, y2, b1, b2 = self.transform([rbc, bg])
        return y1, y2, b1, b2



def prep_rbc_and_bg_data(tif_file, patch_size=1024, goal=10000, check_ditect=True):
    dat_smear = Smear_tiff(tif_file)
    dat_smear.check_dimensions()
    buffer = patch_size/2
    bg_p = 0.01 # 現在は固定

    # patchをランダムに取得するための組み合わせ
    loc_pairs = list(itertools.product(range(0, dat_smear.dimensions[0]//patch_size), range(0, dat_smear.dimensions[1]//patch_size)))
    # ランダムにシャッフルして順番にペアを取得
    random.shuffle(loc_pairs)

    total_image = deque()
    #structural_line_image = deque()
    #blurred_image = deque()

    # ペアを順番に処理
    #n = 0 # errorの判定に使用
    with tqdm(total=goal, desc="total rbc", leave=False, position=1) as pbar:
        for loc_n, xy in enumerate(loc_pairs):
            if loc_n == len(loc_pairs) - 1:
                print("All regions have been searched.")
            x = int(xy[0]*patch_size + buffer)
            y = int(xy[1]*patch_size + buffer)

            isolated_centroids, isolated_areasize = dat_smear.ditect_rbc(patch_size=patch_size, loc=(x, y), rbc_radius=60)
            bg_size = 80
            background_img, bg_x, bg_y = dat_smear.get_background_cor(patch_size=patch_size, loc=(x, y), rbc_size=bg_size, bg_p=bg_p)
            if isolated_centroids is not None and background_img is not None: 
                rbc_lst = dat_smear.get_rbcimage(isolated_centroids, isolated_areasize, loc=(x, y))

                # ======= ここから書き換え ======= #
                for rgb_array in rbc_lst:
                    if laplacian_filter(rgb_array, threshold=3):
                        if detect_structural_line_spike(rgb_array):
                            #structural_line_image.extend([rgb_array])
                            pass
                        else:
                            total_image.extend([[rgb_array, background_img]]) # どちらもnp.array, 
                            # print("RBC shape:", rgb_array.shape, "BG shape:", background_img.shape) -> RBC shape: (128, 128, 3) BG shape: (128, 128, 3)
                            pbar.update(1)
                    else:
                        pass

                # ======= ここまで書き換え ======= #

                if len(total_image) > goal:
                    #print("The goal has been reached.")
                    if check_ditect:
                        image_array = np.array(dat_smear.get_area(patch_size=patch_size, loc=(x, y)), dtype=np.uint8)
                        plt.scatter(isolated_centroids[:,0],isolated_centroids[:,1],s=50, marker='h',c='orangered')
                        ax = plt.gca()
                        # 矩形の座標とサイズ
                        rect = patches.Rectangle(
                            (bg_y, bg_x),              # 左上座標（列, 行）
                            bg_size, bg_size,  # 横幅, 高さ
                            linewidth=2,
                            edgecolor='red',
                            facecolor='none'
                        )

                        # 矩形を追加
                        ax.add_patch(rect)

                        plt.imshow(image_array)#これは縦横 (y, x)
                        plt.show()
                    break
            else:
                dat_smear = Smear_tiff(tif_file)
                pbar.update(0)
                #n = 1
                continue


    total_image = list(total_image)
    #blurred_image = list(blurred_image)
    #structural_line_image = list(structural_line_image)
    total_image = total_image[0:goal]

    if check_ditect:
        show_get_img2(total_image)
        #show_get_img(structural_line_image)
        #show_get_img(blurred_image)

    return total_image


def prep_dataset(total_image, splitn=1):
    if splitn > 1:
        random.shuffle(total_image)
        my_datasets = [SmearDataset_RBC_BG(i) for i in np.array_split(total_image, splitn)]
    else:
        my_datasets = [SmearDataset_RBC_BG(total_image)]

    return my_datasets


def prep_bgcor_dataset(
    image_path, 
    num_rbc=2000, 
    show_imagedata=True, 
    ssl_transform=None, 
    data_save=False, 
    save_path="rbc_bg_dataset.npz"
    )  -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if type(image_path) == str:
        image_paths = [image_path]
    elif type(image_path) == list:
        image_paths = image_path

    np_img_lst = []
    for path in tqdm(image_paths, desc="Processing images", position=0):
        total_image =  prep_rbc_and_bg_data(path, patch_size=1024, goal=num_rbc, check_ditect=show_imagedata)
        np_img_lst.extend(total_image)

    if data_save:
        rbc_images = [pair[0] for pair in np_img_lst]
        bg_images = [pair[1] for pair in np_img_lst]
        rbc_array = np.array(rbc_images)
        bg_array = np.array(bg_images)
        np.savez_compressed(save_path, rbc=rbc_array, bg=bg_array)

    return _create_and_split_dataset(np_img_lst, ssl_transform)

def load_bgcor_dataset(npz_path, ssl_transform=None):
    data = np.load(npz_path)
    rbc_array = data['rbc']
    bg_array = data['bg']

    np_img_lst = [list(pair) for pair in zip(rbc_array, bg_array)]

    return _create_and_split_dataset(np_img_lst, ssl_transform)


def _create_and_split_dataset(
    np_img_lst, 
    ssl_transform: object, 
    split_ratio: float = 0.8, 
    random_seed: int = 24771
    ):
    """
    リストからデータセットを作成し、訓練/テスト用に分割する内部関数
    """
    smeardataset = prep_dataset(np_img_lst, splitn=1)
    mydataset = Dataset_SSL_add_BG(smeardataset, ssl_transform)
    
    dataset_size = len(mydataset)
    train_size = int(split_ratio * dataset_size)
    test_size = dataset_size - train_size

    # --- 分割の再現性を確保 ---
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, test_dataset = random_split(
        mydataset, [train_size, test_size], generator=generator
    )

    print("===============================================================================")
    print("train:test =", str(len(train_dataset)),":", str(len(test_dataset)))
    
    return train_dataset, test_dataset


def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True
    ) -> torch.utils.data.DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
        )    
    return loader


def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prep_smeardata_ss_bgcor(
    load_data:bool=True,
    path=None, 
    num_rbc=2000, 
    batch_size:int=0, 
    ssl_transform=None, 
    shuffle=(True, False),
    num_workers:int=2, 
    pin_memory:bool=True, 
    dataset_save:bool=True,
    save_path="rbc_bg_dataset.npz",
    show_imagedata:bool=True
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if load_data:
        train_dataset, test_dataset = load_bgcor_dataset(path, ssl_transform=ssl_transform)
    else:
        train_dataset, test_dataset = prep_bgcor_dataset(
            path, num_rbc=num_rbc, show_imagedata=show_imagedata, ssl_transform=ssl_transform, data_save=dataset_save, save_path=save_path 
            )

    train_loader = prep_dataloader(
        train_dataset, batch_size, shuffle[0], num_workers, pin_memory
        )
    test_loader = prep_dataloader(
        test_dataset, batch_size, shuffle[1], num_workers, pin_memory
        )    

        
    return train_loader, test_loader



# 以下downstream task用、適宜変更

class Dataset_toViT2(torch.utils.data.Dataset):
    def __init__(self, mydataset):
        self.mydataset = mydataset
        self.datanum = len(self.mydataset)
        self.t = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        input, bg = self.mydataset[idx]
        input = Image.fromarray(input) # 上記判定が不要のため
        bg = Image.fromarray(bg) # 上記判定が不要のため
        img = self.t(input)
        bg = self.t(bg)
        return img, bg
    
def prep_validdataset_bg_lst(image_path, num_image=2000, splitn=1, show_imagedata=True):
    dataset_lst = []
    if type(image_path) == str:
        image_paths = [image_path]
    elif type(image_path) == list:
        image_paths = image_path

    for path in image_paths:
        print(path)
        total_image = prep_rbc_and_bg_data(path, patch_size=1024, goal=num_image, check_ditect=show_imagedata)

        smeardataset = prep_dataset(total_image, splitn=splitn)
        dataset_lst.append((path, smeardataset))

    return dataset_lst


def prep_valid_loader2(mydataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True):
    dataset_tomodel = Dataset_toViT2(mydataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_tomodel,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
        )  
    return dataloader


def get_cls_token_output(model, image_tensor, latent_id="encoder.blocks.2.layernorm2"):
    if latent_id == "encoder.blocks.3.layernorm2":
        with torch.no_grad():
            tokenized_images = model.embedding(image_tensor)
            encoder_output = model.encoder(tokenized_images)[0]  # エンコーダ部の出力を取得
        # CLS トークンは通常、出力の最初のトークンとして存在する
        cls_token = encoder_output[:, 0, :]  # (batch_size, 256)

    elif latent_id == "encoder.blocks.2.layernorm2":
        with torch.no_grad():
            tokenized_images = model.embedding(image_tensor)
            # blocks.0, blocks.1, blocks.2 を順に通過
            x = tokenized_images
            for i in range(3):  # blocks.0, blocks.1, blocks.2 まで処理
                x = model.encoder.blocks[i](x)
                if isinstance(x, tuple):  # tupleだったら1番目だけ使う
                    x = x[0]
            # blocks.2.layernorm2 の出力をそのまま取得
        cls_token = x[:, 0, :]  # ← これでOK！

    return cls_token


def clstoken_extraction_bg(dataloader, model, device=None, f_type="mean+max", latent_id="encoder.blocks.2.layernorm2"):
    # 特徴量を格納するリスト
    features = []
    ap = features.append

    # DataLoader でバッチごとに特徴量を抽出
    for images, bgs in dataloader:
        with torch.no_grad():  # 勾配計算をオフにする
            # バッチをモデルに入力して特徴量を抽出
            images = images.to(device)
            bgs = bgs.to(device)
            cls_token_rbc = get_cls_token_output(model, images, latent_id=latent_id) # (batch_size, 256)
            cls_token_bg = get_cls_token_output(model, bgs, latent_id=latent_id)
            cls_token = cls_token_rbc - cls_token_bg
            ap(cls_token)

    # 全ての特徴量をまとめる
    features = torch.cat(features, dim=0)
    #print(features.shape)  # (総赤血球数, 特徴量の次元数)
    mean_feature = features.mean(dim=0)
    max_feature = features.max(dim=0)[0]
    min_feature = features.min(dim=0)[0]

    if f_type == "mean+max":
        rbc_feature = torch.cat((mean_feature, max_feature), dim=0)
    elif f_type == "mean+max+min":
        rbc_feature = torch.cat((mean_feature, max_feature, min_feature), dim=0)
    elif f_type == "mean":
        rbc_feature = mean_feature
    elif f_type == "max":
        rbc_feature = max_feature
    else:
        raise ValueError("!! Please enter the correct f_type !!")
    #print(rbc_feature.shape)  # torch.Size([256])

    return rbc_feature


def get_clstoken(dataset_lst, model, device, f_type="mean+max", latent_id="encoder.blocks.2.layernorm2", batch_size=64):
    features = {} # 特徴量を辞書形式でまとめる
    model.to(device)
    for n, mydatasets in dataset_lst:
        for i, mydataset in enumerate(mydatasets):
            dataloader = prep_valid_loader2(mydataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            # 学習したViTモデルの読み込み
            model.eval()  # 評価モードに設定（勾配計算をオフに）
            smear_feature = clstoken_extraction_bg(dataloader, model, device=device, f_type=f_type, latent_id=latent_id)
            features[str(n)+"_"+str(i+1)] = smear_feature.cpu() # 後の操作のためにCPUに戻す
        #print("\n")
    sorted_features = {k: features[k] for k in sorted(features)}

    return sorted_features


def get_dr_feature(sorted_features):
    df = pd.DataFrame(sorted_features).T

    # Standardization
    dfs = df.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    dfs_fix = dfs.dropna(axis=1)

    pca = PCA()
    pca.fit(dfs_fix)
    pca_feature = pca.transform(dfs_fix)

    sample_ind = df.index

    return pca_feature, sample_ind



def dr_viz_v3(pca_feature, sample_ind, new_dict, title="PCA (RBC, 24h)", save_dir=None): # ここは必要に応じて変更
    fig, ax = plt.subplots()
    xnum = 0
    ynum = 1

    #ラベルチェック用
    sample_lst = []
    ap = sample_lst.append

    for n, ind in enumerate(sample_ind):
        x, y = pca_feature[:, xnum][n], pca_feature[:, ynum][n]

        if new_dict[ind.split("_")[0]] == "ctrl":
            c = "grey"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

        elif new_dict[ind.split("_")[0]] == "taa":
            c = "orangered"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

        elif new_dict[ind.split("_")[0]] == "mda":
            c = "royalblue"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

        elif new_dict[ind.split("_")[0]] == "apap":
            c = "brown"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

        elif new_dict[ind.split("_")[0]] == "phbr":
            c = "lightseagreen"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

        elif new_dict[ind.split("_")[0]] == "alp":
            c = "green"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

        elif new_dict[ind.split("_")[0]] == "ccl4":
            c = "darkviolet"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

        elif new_dict[ind.split("_")[0]] == "anit":
            c = "turquoise"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

        elif new_dict[ind.split("_")[0]] == "col":
            c = "lime"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

        elif new_dict[ind.split("_")[0]] == "cis":
            c = "darkgreen"
            if new_dict[ind.split("_")[0]] not in sample_lst:
                ax.scatter(x, y, alpha=1, color=c, label=new_dict[ind.split("_")[0]])
                ap(new_dict[ind.split("_")[0]])
            else:
                ax.scatter(x, y, alpha=1, color=c)

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # scatter() のみ凡例に入る
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("PC "+str(xnum+1), fontsize=14)
    ax.set_ylabel("PC "+str(ynum+1), fontsize=14)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir, dpi=300)
    plt.show()


def pseudo_f_statistic(features, labels):
    """
    features: 次元削減後の特徴量 (N x d の numpy array)
    labels: 各サンプルのラベル (N の numpy array)
    """
    N, d = features.shape
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    overall_mean = np.mean(features, axis=0)

    # クラスタごとの重心とサイズ
    cluster_means = []
    cluster_sizes = []
    for lab in unique_labels:
        cluster_points = features[labels == lab]
        cluster_means.append(cluster_points.mean(axis=0))
        cluster_sizes.append(len(cluster_points))
    cluster_means = np.array(cluster_means)
    cluster_sizes = np.array(cluster_sizes)

    # SSB: クラスタ間平方和
    ssb = np.sum(cluster_sizes[:, None] * (cluster_means - overall_mean)**2)

    # SSW: クラスタ内平方和
    ssw = 0
    for lab, mean in zip(unique_labels, cluster_means):
        cluster_points = features[labels == lab]
        ssw += np.sum((cluster_points - mean)**2)

    # Pseudo-F
    pseudo_f = (ssb / (k - 1)) / (ssw / (N - k))

    return pseudo_f


def ds_task(dataset_lst, filename_label_dict, model, device, f_type, latent_id, batch_size, run):
    sorted_features = get_clstoken(dataset_lst, model, device, f_type=f_type, latent_id=latent_id, batch_size=batch_size)
    pca_feature, sample_ind = get_dr_feature(sorted_features)

    sample_ind = [i.split(".")[0].split("-")[-1] + i.split("tif")[1] for i in sample_ind]
    new_dict = {k.split("-")[-1].split(".")[0] : filename_label_dict[k] for k in filename_label_dict.keys()}
    labels = [new_dict[i.split("_")[0]] for i in sample_ind]

    pseudo_f = pseudo_f_statistic(pca_feature, np.array(labels))
    print("Pseudo-F:", pseudo_f)
    
    dbi = davies_bouldin_score(pca_feature, labels)
    print("DBI:", dbi)

    dr_viz_v3(pca_feature, sample_ind, new_dict, title="PCA (RBC, 24h)", save_dir=None)

    run.log({
    "Pseudo-F:": pseudo_f,
    "DBI:": dbi
    })





# ===================================================== #
def laplacian_filter(rgb_array, threshold=10):
    # RGB → グレースケール
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

    # ラプラシアンフィルタ
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # シャープネス（ピントの指標）
    sharpness = laplacian.var()

    return sharpness > threshold

def detect_structural_line_spike(img_rgb, diff_thresh=5, window_ratio=0.2):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # プロファイル取得
    col_profile = np.mean(gray, axis=0)
    row_profile = np.mean(gray, axis=1)

    col_diff = np.abs(np.diff(col_profile))
    row_diff = np.abs(np.diff(row_profile))

    # スパイクが画像の中央付近に集中してるかをチェック
    col_center = w // 2
    row_center = h // 2
    win_c = int(w * window_ratio)
    win_r = int(h * window_ratio)

    vertical_spike = np.max(col_diff[col_center - win_c: col_center + win_c]) > diff_thresh
    horizontal_spike = np.max(row_diff[row_center - win_r: row_center + win_r]) > diff_thresh

    return vertical_spike or horizontal_spike