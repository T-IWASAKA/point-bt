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
from collections import deque, defaultdict
from typing import Tuple
import os
import sys
import cv2
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import random_split
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

from openslide import OpenSlide, OpenSlideError
from scipy.spatial import KDTree

from .models.point_bt import Image2Feature


class SmearDataset_RBC(torch.utils.data.Dataset):
    """
    画像をメモリに読み込む高速版
    メモリ効率化のため、PILオブジェクトではなくnumpy配列(uint8)で保持する
    """
    def __init__(self, image_lst, transform=None):
        self.transform = transform
        self.images = [] 

        print(f"Loading {len(image_lst)} images to memory...")
        
        for image_path in tqdm(image_lst):
            # 1. 画像を開いてRGB変換
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                # 2. ここで numpy array (uint8) に変換してリストに入れる
                #    これによりファイルハンドルは確実に閉じられ、メモリもコンパクトになる
                self.images.append(np.array(img))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 3. 取り出す時に PIL Image に戻す
        #    (多くのTorchvision TransformsはPIL入力を期待するため)
        img_array = self.images[idx]
        image_rbc = Image.fromarray(img_array)

        # ここではtransformは適用しない（Dataset_SSL側でやるため）
        # もしこのクラス単体で使うならここで self.transform(image_rbc) する
        
        return image_rbc


class Dataset_SSL(torch.utils.data.Dataset): 
    """
    SSL用: 1枚の画像から2つのAugmentationビューを作る
    """
    def __init__(self, mydataset, transform):
        if transform is None:
            raise ValueError('!! Give transform !!')
        self.transform = transform
        self.input = mydataset 

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        # input (SmearDataset_RBC) から PIL Image が返ってくる
        rbc = self.input[idx]
        
        # transformで2枚の画像(y1, y2)に変換
        y1, y2 = self.transform(rbc)
        return y1, y2


# ===================== 書き換えた ===================== #

def get_patch_data(os_obj, loc, patch_size=1024):
    """ 画像と二値マスクを取得 """
    try:
        wsi_img = os_obj.read_region(location=loc, level=0, size=(patch_size, patch_size)).convert("RGB")
        img_arr = np.array(wsi_img, dtype=np.uint8)
        del wsi_img
    except Exception:
        return None, None

    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    return img_arr, binary

def extract_rbc_crops(img_arr, binary, patch_size=1024, rbc_radius=60, target_size=(128, 128)):
    """ 孤立した赤血球を切り出す """
    if img_arr is None: return [], []

    # 連結成分抽出
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1: return [], []

    # 背景(label=0)除外
    centroids = centroids[1:]
    stats = stats[1:]
    areas = stats[:, 4]

    # 1. 高速フィルタリング (サイズ & 境界)
    min_area = rbc_radius * rbc_radius / 3.5
    max_area = rbc_radius * rbc_radius * 2
    margin = 50
    
    valid_mask = (areas > min_area) & (areas < max_area) & \
                 (centroids[:, 0] > margin) & (centroids[:, 0] < patch_size - margin) & \
                 (centroids[:, 1] > margin) & (centroids[:, 1] < patch_size - margin)
    
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0: return [], []

    candidate_centroids = centroids[valid_indices]
    candidate_areas = areas[valid_indices]

    # 2. KDTreeによる孤立点判定
    tree = KDTree(candidate_centroids)
    neighbors_lst = tree.query_ball_point(candidate_centroids, rbc_radius)
    is_isolated = [len(n) == 1 for n in neighbors_lst]

    final_centroids = candidate_centroids[is_isolated]
    final_areas = candidate_areas[is_isolated]

    # 3. 画像切り出し
    rbc_crops = []
    pad = int(rbc_radius * 3)
    img_padded = cv2.copyMakeBorder(img_arr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))

    for (cx, cy), area in zip(final_centroids, final_areas):
        radius = np.sqrt(area / np.pi)
        crop_r = int(1.03 * radius)
        
        x1, y1 = int(cx + pad - crop_r), int(cy + pad - crop_r)
        x2, y2 = int(cx + pad + crop_r), int(cy + pad + crop_r)
        
        crop = img_padded[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
        rbc_crops.append(crop_resized)

    return rbc_crops, final_centroids

def create_dataset_from_wsi(tif_file, patch_size=1024, goal=10000, save_dir="", check_detect=True):
    """
    WSIから赤血球画像のみを収集する
    Return: [img1, img2, ...] (ペアではなく画像のパスのリスト)
    """
    try:
        os_obj = OpenSlide(tif_file)
    except Exception as e:
        print(f"File Open Error: {e}")
        return []

    dims = os_obj.dimensions
    buffer = patch_size / 2
    
    loc_pairs = list(itertools.product(
        range(0, dims[0] // patch_size), 
        range(0, dims[1] // patch_size)
    ))
    random.shuffle(loc_pairs)

    save_paths = deque()
    count = 0
    with tqdm(total=goal, desc="Collecting RBCs", leave=False, position=1) as pbar:
        for loc_n, xy in enumerate(loc_pairs):
            if len(save_paths) >= goal: break
                
            x = int(xy[0] * patch_size + buffer)
            y = int(xy[1] * patch_size + buffer)

            # 画像読み込み & RBC抽出
            img_arr, binary = get_patch_data(os_obj, (x, y), patch_size)
            if img_arr is None: continue
            
            rbc_crops, centroids = extract_rbc_crops(
                img_arr, binary, patch_size=patch_size, rbc_radius=60
            )
            
            if not rbc_crops: continue

            # フィルタリング & 保存
            for rbc_img in rbc_crops:
                # ※ 外部フィルタ関数
                if laplacian_filter(rbc_img, threshold=3):
                    if detect_structural_line_spike(rbc_img):
                        pass # ゴミ
                    else:
                        filename = f"{count:06}.png"
                        # ==== 接頭辞はAxioScanに合わせたハードコード ====
                        filename_only = os.path.basename(tif_file)
                        filename_head = filename_only.split("_")[0]+filename_only.split("_")[1]+filename_only.split("_")[2]+filename_only.split(".")[0].split("-")[-1]
                        # =============================================
                        save_path = os.path.join(save_dir, filename_head+"_"+filename) # _で区切ることで後のPointNetの側の群分けを可能にする、ここもハードコード
                        img_bgr = cv2.cvtColor(rbc_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(save_path, img_bgr)
                        save_paths.append(save_path) # 画像pathを追加
                        pbar.update(1)
                        count += 1
                        if count >= goal: break
                else:
                    pass # ボケ

            # 確認用可視化 (ゴール達成時のみ)
            if len(save_paths) >= goal and check_detect:
                _visualize_check(img_arr, centroids, len(save_paths))
                break

    os_obj.close()
    
    if check_detect and len(save_paths) > 0:
        print(f"Collected {len(save_paths)} images.")
        _show_sample_grid(list(save_paths)[:16])

    return list(save_paths)


def _visualize_check(img_arr, centroids, count):
    """ 検出位置の確認 """
    plt.figure(figsize=(8, 8))
    plt.imshow(img_arr)
    if len(centroids) > 0:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=50, marker='h', c='orangered')
    plt.title(f"Check: {count} images collected")
    plt.show()

def _show_sample_grid(image_paths):
    """ 収集した画像のグリッド表示 (単体画像用) """
    if not image_paths: return
    num = len(image_paths)
    rows = (num - 1) // 4 + 1
    fig = plt.figure(figsize=(8, rows * 2))
    for i, path in enumerate(image_paths):
        ax = fig.add_subplot(rows, 4, i + 1)
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ===================== 書き換えた ===================== #

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

# ===================================================== #

def prep_dataset(total_image, splitn=1):
    if splitn > 1:
        random.shuffle(total_image)
        my_datasets = [SmearDataset_RBC(i) for i in np.array_split(total_image, splitn)]
    else:
        my_datasets = [SmearDataset_RBC(total_image)]

    return my_datasets


def prep_imagepath_dataset(
    image_path, 
    num_rbc=2000, 
    show_imagedata=True, 
    ssl_transform=None, 
    save_path="data/save/path"
    )  -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if len(glob.glob(os.path.join(save_path, "*.png"))) > 0:
        np_img_lst = glob.glob(os.path.join(save_path, "*.png"))
    else:
        np_img_lst = []
        for path in tqdm(image_path, desc="Processing images", position=0):
            total_image =  create_dataset_from_wsi(path, patch_size=1024, goal=num_rbc, save_dir=save_path, check_detect=show_imagedata)
            np_img_lst.extend(total_image)

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
    mydataset = Dataset_SSL(smeardataset[0], ssl_transform)
    
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


def prep_smeardata_ssl(
    path=None, 
    num_rbc=2000, 
    batch_size:int=0, 
    ssl_transform=None, 
    shuffle=(True, False),
    num_workers:int=2, 
    pin_memory:bool=True, 
    save_path="data/save/path",
    show_imagedata:bool=True
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset, test_dataset = prep_imagepath_dataset(
        path, num_rbc=num_rbc, show_imagedata=show_imagedata, ssl_transform=ssl_transform, save_path=save_path 
        )

    train_loader = prep_dataloader(
        train_dataset, batch_size, shuffle[0], num_workers, pin_memory
        )
    test_loader = prep_dataloader(
        test_dataset, batch_size, shuffle[1], num_workers, pin_memory
        )    

        
    return train_loader, test_loader


# 以下pointbt用、適宜変更 ==============================================================


# === 1. 推論用のデータセット定義 ===
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, resize=(64, 64)):# resizeは後で外からさわれるようにする、具体的にはcropサイズに変更
        self.file_paths = file_paths
        self.transform = transforms.Compose([
            transforms.Resize(resize), # PILのままリサイズの方が速いことが多い
            transforms.ToTensor(),     # (C, H, W) になり、0-1に正規化される
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with Image.open(path) as i:
            pil_img = i.convert('RGB')
        
        img_tensor = self.transform(pil_img)
        # ファイル名も返して、保存時に使う
        filename = os.path.basename(path)
        return img_tensor, filename

# === 2. メインの変換関数 ===
def image2feature(data_path, save_dir, backbone_nn, latent_id, backbone_projector, batch_size=256, device="cuda"):
    os.makedirs(save_dir, exist_ok=True)
    
    # ファイルリスト取得
    all_files = glob.glob(os.path.join(data_path, "*.png"))
    if not all_files:
        print("画像が見つかりません！")
        return

    # Dataset & DataLoader (高速化の鍵)
    dataset = InferenceDataset(all_files, resize=(64, 64))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # モデルの準備
    rbc_encoder = Image2Feature(backbone_nn, latent_id=latent_id, projector=backbone_projector)
    rbc_encoder.to(device)
    rbc_encoder.eval()  # ★重要: 評価モード

    print(f"Start processing {len(all_files)} images...")

    # ★重要: 勾配計算なしでメモリ節約
    with torch.no_grad():
        for images, filenames in tqdm(loader):
            images = images.to(device) # (Batch, C, H, W)

            # 推論 (Batch分まとめて計算)
            features = rbc_encoder(images) # (Batch, Feature_Dim)

            # GPUからCPUへ戻し、Numpy化
            features = features.cpu().numpy().astype(np.float32) # ★重要: float32

            # 1個ずつ保存
            for i, filename in enumerate(filenames):
                # 特徴量ベクトルを取り出す
                feat_vec = features[i] 
                
                # 保存パス
                save_name = os.path.splitext(filename)[0] + ".npz"
                save_path = os.path.join(save_dir, save_name)
                
                # 保存
                np.savez_compressed(save_path, feats=feat_vec) # ここのfeatsをprep_pointnet_dataset内のキーと一致させる必要あり

    print("完了！")
    

class PointDataset_RBC(torch.utils.data.Dataset):
    """
    特徴量ベクトル(npz)を扱う軽量版データセット
    画像処理は事前に行われているため、ここでは単にデータを返すだけ
    """
    def __init__(self, feature_pair_lst):
        """
        Args:
            feature_pair_lst: List[Tuple[np.ndarray, np.ndarray]]
            ペアになった特徴量配列のリスト。
            各要素は ( (N, Dim), (N, Dim) ) の形になっていることを想定。
        """
        self.pairs = feature_pair_lst
        print(f"Dataset initialized with {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        # 既に (N, Dim) の numpy array になっているので、Tensorにするだけ
        feat_set1, feat_set2 = self.pairs[idx]
        
        # float32に変換
        set1_tensor = torch.from_numpy(feat_set1).float()
        set2_tensor = torch.from_numpy(feat_set2).float()
        
        return set1_tensor, set2_tensor

def prep_pointnet_dataset(
    save_path="data/rbc_features", # npzの保存場所
    n_rbc_set=100,
    expansion_factor=10, # ★追加: データを何倍に拡張するか (疑似Augmentation)
    split_ratio=0.8,
    random_seed=24771
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    
    os.makedirs(save_path, exist_ok=True)
    feature_pair_lst = []

    # npzファイルを探す
    all_files = glob.glob(os.path.join(save_path, "*.npz"))
    
    # === A. 既に特徴量(npz)がある場合 ===
    if len(all_files) > 0:
        print(f"Found {len(all_files)} feature files. Loading...")
        
        # 1. スライドIDごとにファイルをグルーピング
        slide_dict = defaultdict(list)
        for path in tqdm(all_files, desc="Grouping by Slide ID"):
            filename = os.path.basename(path)
            # ファイル名規則: "SlideID_....npz" を想定
            slide_id = filename.split("_")[0]
            slide_dict[slide_id].append(path)

        # 2. 各スライドごとにデータを読み込み、拡張しながらペアを作成
        for slide_id, file_paths in tqdm(slide_dict.items(), desc="Creating Augmented Pairs"):
            
            # --- ここでスライド内の全特徴量をメモリにロード ---
            # I/O回数を減らすため一気に読み込む
            slide_features = []
            for p in file_paths:
                try:
                    # np.loadは辞書形式を返すので、キー'feats'を指定
                    # (前のコードで save_compressed(..., feats=...) とした想定)
                    with np.load(p) as data:
                        slide_features.append(data['feats'])
                except Exception as e:
                    print(f"Error loading {p}: {e}")
                    continue
            
            if not slide_features:
                continue
                
            # numpy arrayのリストになっているので、取り扱いやすいようにstackするかリストのままで
            # ここではリストのまま扱います (要素は (Dim,) のベクトル)
            
            total_rbcs = len(slide_features)
            imgs_per_pair = n_rbc_set * 2
            
            # 特徴量が少なすぎてペアが作れない場合はスキップ
            if total_rbcs < imgs_per_pair:
                continue

            # ★★★ 疑似Augmentation (Shuffle & Repeat) ★★★
            # expansion_factor の回数だけ、シャッフルして異なる組み合わせを作る
            for _ in range(expansion_factor):
                # プールをコピーしてシャッフル
                current_pool = slide_features.copy()
                random.shuffle(current_pool)
                
                # ペア作成数
                num_pairs = len(current_pool) // imgs_per_pair
                
                for i in range(num_pairs):
                    # 前半セット (N, Dim)
                    set1 = np.stack(current_pool[i * imgs_per_pair : i * imgs_per_pair + n_rbc_set])
                    # 後半セット (N, Dim)
                    set2 = np.stack(current_pool[i * imgs_per_pair + n_rbc_set : (i + 1) * imgs_per_pair])
                    
                    feature_pair_lst.append((set1, set2))

    # === B. npzがない場合 (まだ特徴量抽出していない) ===
    else:
        raise FileNotFoundError(
            f"No .npz files found in {save_path}. \n"
            "Please run 'image2feature' function first to generate feature vectors!"
        )

    print(f"Total pairs generated: {len(feature_pair_lst)} (Expansion: x{expansion_factor})")
    
    # データセット作成
    return _create_point_dataset(feature_pair_lst, split_ratio, random_seed)


def _create_point_dataset(
    img_pair__lst, 
    split_ratio: float = 0.8, 
    random_seed: int = 24771
    ):
    """
    リストからデータセットを作成し、訓練/テスト用に分割する内部関数
    """
    mydataset = PointDataset_RBC(img_pair__lst)
    
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


def prep_smeardata_pointbt( #prep_smeardata_sslと同様外に持っていく関数
    save_path="data/rbc_features",
    n_rbc_set=100, 
    expansion_factor=10,
    batch_size:int=1, 
    shuffle=(True, False),
    num_workers:int=2, 
    pin_memory:bool=True, 
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset, test_dataset = prep_pointnet_dataset(
        save_path=save_path, n_rbc_set=n_rbc_set, expansion_factor=expansion_factor
        )

    train_loader = prep_dataloader(
        train_dataset, batch_size, shuffle[0], num_workers, pin_memory
        )
    test_loader = prep_dataloader(
        test_dataset, batch_size, shuffle[1], num_workers, pin_memory
        )    

        
    return train_loader, test_loader



# 以下downstream task用、適宜変更 ==============================================================

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