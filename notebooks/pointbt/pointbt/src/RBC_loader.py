"""
Created on 2025-02-23

@author: T.Iwasaka
"""

import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openslide import OpenSlide, OpenSlideError
from scipy.spatial import KDTree
from torchvision import transforms

class Smear_tiff():
    def __init__(
        self,
        slide_path,
    ):
        self.OS = OpenSlide(slide_path)
        self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(128, 128)), # 現在は固定, 必要に応じて外部指定可能にしてもよい, こいつはbgのResizeと揃える必要あり
                ]
            )

    def check_dimensions(self):
        # load WSI
        dims = self.OS.dimensions
        self.dimensions = dims
        #print("(x, y) =", str(self.dimensions))
    
    def get_area(self, patch_size=1024, loc=(17010, 45700)):
        wsi_test = self.OS.read_region(location=loc, level=0, size=(patch_size,patch_size)).convert("RGB")
        return wsi_test

    def ditect_rbc(self, patch_size=1024, loc=(17010, 45700), rbc_radius=60):
        # 全体のバイナリイメージ
        try:
            wsi_test = self.OS.read_region(location=loc, level=0, size=(patch_size,patch_size)).convert("RGB")
        except OpenSlideError:
            return None, None # continueに繋がるようにNoneを返す
        
        image_array = np.array(wsi_test, dtype=np.uint8)
        del wsi_test
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) #OTSU
        del gray
        del ret

        # connectedComponentsWithStatsで領域ごとに抽出
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        # KDtreeを用いて重心同士の距離が近いものを除外
        tree = KDTree(centroids[1:])
        
        # 各点について、半径60以内に他の点がないか調べる
        isolated_centroids = [] # 重心の制約+面積が大きすぎるor小さすぎるものを除外
        ap_point = isolated_centroids.append
        isolated_areasize = [] # 領域サイズも取得しておく
        ap_area = isolated_areasize.append

        for label in range(1, num_labels): # 背景(label=0)を除外
            point = centroids[label] # 重心座標 (x, y)
            area = stats[label, 4]  # 領域のピクセル数
            indices = tree.query_ball_point(point, rbc_radius)
            if (point[0] < 50 or patch_size - point[0] < 50) or (point[1] < 50 or patch_size - point[1] < 50): #パッチの端にある点は除外
                pass
            elif len(indices) == 1:  # 自分自身しか含まれないなら孤立点
                if rbc_radius*rbc_radius/4 < area < rbc_radius*rbc_radius*2: # 小さすぎるまたは大きすぎるものを除外
                    ap_point(point)
                    ap_area(area)
            else:
                pass

        # 結果をnumpy配列に変換
        isolated_centroids = np.array(isolated_centroids) #これは横縦 (x, y)
        isolated_areasize = np.array(isolated_areasize)

        return isolated_centroids, isolated_areasize
    
    def get_rbcimage(self, isolated_centroids, isolated_areasize, loc=(17010, 45700)):
        rbc_lst = []
        ap = rbc_lst.append
        for rbc_loc, areasize in zip(isolated_centroids, isolated_areasize):
            radius = np.sqrt(areasize / np.pi)
            new_loc = (int(loc[0]+rbc_loc[0]-radius), int(loc[1]+rbc_loc[1]-radius))
            try:
                rbc_area = self.OS.read_region(location=new_loc, level=0, size=(int(1.03*radius*2),int(1.03*radius*2))).convert("RGB")
            except:
                continue
            # ここで画像を一括で変換するため、サイズ感は失われる
            image_transform = self.transform(rbc_area) #resize
            image_np = np.array(image_transform)[:,:,:3]
            del rbc_area
            del image_transform
            ap(image_np)
        
        return rbc_lst
    
    def get_background(self, patch_size=1024, loc=(17010, 45700), rbc_size=60, bg_p=0.01):
        # 全体のバイナリイメージ
        try:
            wsi_test = self.OS.read_region(location=loc, level=0, size=(patch_size,patch_size)).convert("RGB")
        except OpenSlideError:
            return [] # continueに繋がるように[]を返す
        image_array = np.array(wsi_test, dtype=np.uint8)
        del wsi_test
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) #OTSU
        del gray
        del ret

        background_lst = []
        ap = background_lst.append

        for x in range(0, patch_size, rbc_size):
            for y in range(0, patch_size, rbc_size):
                if int(np.sum((binary/255)[x:x+rbc_size, y:y+rbc_size])) < (rbc_size**2)*bg_p: # ここの値次第で変わる
                    ap(image_array[x:x+rbc_size, y:y+rbc_size])

        return background_lst
    

    def get_background_cor(self, patch_size=1024, loc=(17010, 45700), rbc_size=60, bg_p=0.01):
        # 全体のバイナリイメージ
        try:
            wsi_test = self.OS.read_region(location=loc, level=0, size=(patch_size,patch_size)).convert("RGB")
        except OpenSlideError:
            return None, None, None # continueに繋がるようにNoneを返す
        image_array = np.array(wsi_test, dtype=np.uint8)
        del wsi_test
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) #OTSU
        del gray
        del ret
        
        return_image, x, y = None, None, None

        for x in range(0, patch_size, rbc_size):
            for y in range(0, patch_size, rbc_size):
                if int(np.sum((binary/255)[x:x+rbc_size, y:y+rbc_size])) < (rbc_size**2)*bg_p: # ここの値次第で変わる
                    return_image = image_array[x:x+rbc_size, y:y+rbc_size]
                    return_image = cv2.resize(return_image, (128, 128)) # rbcのResizeと揃える
                    return return_image, x, y
                
        return return_image, x, y

def show_get_img(total_image):
    num = 1000
    fig = plt.figure(figsize=(4, (((num-1)//4)+1)))
    n = 1
    for img in random.sample(total_image, len(total_image)): # ランダムに取り出したい
        ax = fig.add_subplot((((num-1)//4)+1), 4, n)
        ax.imshow(img)
        ax.axis('off')
        if n == 16:
            plt.tight_layout()
            plt.show()
            break
        n = n + 1

def show_get_img2(total_image):
    num = 1000
    fig = plt.figure(figsize=(4, (((num-1)//4)+1)))
    n = 1
    for img in total_image: # ランダムに取り出したくない
        ax = fig.add_subplot((((num-1)//4)+1), 4, n)
        ax.imshow(img[0])
        ax.axis('off')
        if n == 16:
            plt.tight_layout()
            plt.show()
            break
        n = n + 1
        ax2 = fig.add_subplot((((num-1)//4)+1), 4, n)
        ax2.imshow(img[1])
        ax2.axis('off')
        if n == 16:
            plt.tight_layout()
            plt.show()
            break
        n = n + 1