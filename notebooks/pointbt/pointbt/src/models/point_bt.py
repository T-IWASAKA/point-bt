# -*- coding: utf-8 -*-
"""
Created on Nov 28 2025

Barlow Twins

models file

@author: iwasaka14
"""
import torch
import torch.nn as nn


class NetWrapper(nn.Module):
    """ inspired by https://github.com/lucidrains/byol-pytorch """
    def __init__(self, net, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.hidden = None
        self.hook_registered = False


    def _find_layer(self):
        if type(self.layer)==str: # 名称で取得
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer)==int: # indexで取得
            children = [*self.net.children()]
            return children[self.layer]
        return None


    def _hook(self, _, __, output):
        self.hidden = output


    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"!! Hidden layer ({self.layer}) not found !!"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True


    def get_representation(self, x):
        if self.layer==-1:
            return self.net(x)
        if not self.hook_registered:
            self._register_hook()
        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None # self.hiddenを初期化している
        assert hidden is not None, f"!! Hidden layer ({self.layer}) never emitted an output !!"
        return hidden


    def forward(self, x):
        representation = self.get_representation(x)
        #print(representation.shape)
        representation = representation[:, 0, :] # CLSトークンである一番上を取得

        return representation


def off_diagonal(x):
    """ return a flattened view of the off-diagonal elements of a square matrix """
    n, m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Image2Feature(nn.Module):
    """

    """
    def __init__(self, backbone_nn, latent_id=None, projector=None):
        """
        Parameters
        ----------
        backbone_nn: Model

        latent_id: name or index of the layer to be fed to the pointbt

        projector: the projection head used during training

        """
        super().__init__()
        # VitForPointBTを使えばどちらもbackbornからの出力はCLSトークン
        self.backbone_nn = backbone_nn
        if latent_id:
            self.backbone_nn = NetWrapper(self.backbone_nn, latent_id)
        # projector, 出力の次元に注意
        if projector:
            self.projector = projector
        else:
            self.projector = nn.Identity()

    def forward(self, y): # 画像群の平坦化されたものを入力 ((batch x n(defaultでは100)) x image)
        z = self.backbone_nn(y)
        z = self.projector(z)
        return z # b*n x hidden (or b*n x projector_size)


class RBCPointNet(nn.Module):
    """

    """
    def __init__(self, input_dim, output_dim):
        """
        Parameters
        ----------
        backbone_nn: Model

        latent_id: name or index of the layer to be fed to the pointbt

        projector: the projection head used during training

        """
        super().__init__()
        # ポイント1: kernel_size=1 の Conv1d
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, ys): # 画像群の特徴量化されたものを入力((batch x n(defaultでは100)) x image)
        # 次元の入れ替え (Transpose)
        ys = ys.transpose(1, 2) 
        # ys の形: [Batch, input_dim, n]

        ys = self.mlp(ys) # 各細胞ごとの独立した変換 (MLP)
        # x の形: [Batch, output_dim, n]

        ys = torch.max(ys, 2)[0] 
        # x の形: [Batch, output_dim]
        
        return ys


class PointBT(nn.Module):
    """
    single GPU version based on https://github.com/facebookresearch/barlowtwins

    """
    def __init__(self, point_input_dim, projection_sizes, lambd, scale_factor=1):
        """
        Parameters
        ----------
        backbone_nn: Model

        latent_id: name or index of the layer to be fed to the projection

        projection_sizes: size of the hidden layers in the projection

        lambd: tradeoff function

        scale_factor: factor to scale loss by

        """
        super().__init__()
        point_output_dim = projection_sizes[0]
        self.pointnet = RBCPointNet(point_input_dim, point_output_dim)
        self.lambd = lambd
        self.scale_factor = scale_factor
        # projector
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False)) # BatchNorm入れるのでbias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False)) # BatchNorm入れるのでbias=False
        self.projector = nn.Sequential(*layers)
        # normalization layer for z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)


    def forward(self, y1s, y2s): # 2つの特徴量群を入力
        # y1s, y2s の入力形状: (Batch, n, hidden)
        # 点群データのBT
        z1 = self.pointnet(y1s)
        z2 = self.pointnet(y2s)
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2))
        c.div_(z1.shape[0])
        # scaling
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)

        return loss, on_diag, off_diag
    

# 以下特徴量化を含めて学習する際に使う、VRAMの容量的にしんどいか
class PointBT_old(nn.Module):
    """
    single GPU version based on https://github.com/facebookresearch/barlowtwins

    """
    def __init__(self, backbone_nn, latent_id, backbone_projector, point_input_dim, projection_sizes, lambd, scale_factor=1):
        """
        Parameters
        ----------
        backbone_nn: Model

        latent_id: name or index of the layer to be fed to the projection

        projection_sizes: size of the hidden layers in the projection

        lambd: tradeoff function

        scale_factor: factor to scale loss by

        """
        super().__init__()
        self.rbc_encoder = Image2Feature(backbone_nn, latent_id=latent_id, projector=backbone_projector)
        self.backbone = self.rbc_encoder.backbone_nn
        point_output_dim = projection_sizes[0]
        self.pointnet = RBCPointNet(point_input_dim, point_output_dim)
        self.lambd = lambd
        self.scale_factor = scale_factor
        # projector
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False)) # BatchNorm入れるのでbias=False
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False)) # BatchNorm入れるのでbias=False
        self.projector = nn.Sequential(*layers)
        # normalization layer for z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)


    def forward(self, y1s, y2s): # 2つの画像群を入力 (y1, y2はそれぞれ batch x n(defaultでは100) x image(hxwxc))
        # y1s, y2s の入力形状: (Batch, n, H, W, C)
        # 1. ここで permute を入れて (B, n, C, H, W) に直す
        y1s = y1s.permute(0, 1, 4, 2, 3) 
        y2s = y2s.permute(0, 1, 4, 2, 3)
        # 特徴量化
        B, n, C, H, W = y1s.shape # 入力の形を取得、元に戻すときに使う
        y1s = y1s.flatten(0, 1)
        y2s = y2s.flatten(0, 1)
        z1s = self.rbc_encoder(y1s)
        z2s = self.rbc_encoder(y2s)
        z1s = z1s.view(B, n, -1) # 入力の形に戻す
        z2s = z2s.view(B, n, -1) # 入力の形に戻す

        # 点群データのエンコード
        z1 = self.pointnet(z1s)
        z2 = self.pointnet(z2s)

        # 点群データのBT
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2))
        c.div_(z1.shape[0])
        # scaling
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)

        return loss, on_diag, off_diag