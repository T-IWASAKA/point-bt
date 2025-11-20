# template-dev
研究開発用テンプレート

AI/MLプロジェクトを開始するための標準開発テンプレート。
日常の研究開発における効率性とコードの整理しやすさの両立を目的とする。

## 特徴

  - **`pyproject.toml` による依存関係の一元管理**
  - **安全な `src` レイアウトの採用**
  - **編集可能インストール (`pip install -e .`) によるスムーズな開発**

## ディレクトリ構成

```
.
├── data/             # データ格納ディレクトリ
├── notebooks/        # 試行錯誤用Jupyter Notebook
├── src/
│   └── my_project/   # Pythonパッケージのソースコード
│       └── __init__.py
├── .gitignore
├── pyproject.toml    # プロジェクト定義ファイル
└── README.md         # 説明書
```

## 利用手順

### 1\. リポジトリの作成

GitHub上で "Use this template" ボタンを押し, 新規リポジトリを作成する。

### 2\. 環境のセットアップ

ローカルにクローン後, 以下のコマンドを実行する。

```bash
# clone
git clone -b {branch名} {URL}
cd {repository名}
```
基本インストール (開発環境が整っているコンテナではこれでOK)。

```bash
# 編集可能モードでインストールすることでsrc以下の編集が即座に反映される
pip install -e "."
```

開発用ツールも含めたフルインストールの場合は以下 (詳細はtoml参照)
```bash
pip install -e ".[dev]"
```

### 3\. プロジェクト名の設定

1.  `pyproject.toml` 内の `name` を変更する。
2.  `src/my_project` ディレクトリ名を `pyproject.toml` の `name` と一致させる。

## 開発フロー

1.  再利用可能なコードは `src/` 以下に記述する。
2.  実験や分析は `notebooks/` で行う。
3.  ノートブックからは, `from my_project import ...` のように自作モジュールを直接インポートして使用できる。
  
***
***
***
# ▼ テンプレート利用時は上記を全て削除し, 以下をプロジェクトに合わせて編集する ▼
***

# {project_name}
short description  

## Note
This repository is under construction and will be officially released by [Mizuno group](https://github.com/mizuno-group).  
Please contact tadahaya[at]gmail.com before publishing your paper using the contents of this repository.  

## Directory Structure
```
.
├── data/             # data directory
├── notebooks/        # Jupyter Notebook
├── src/
│   └── my_project/   # reusable codes
│       └── __init__.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Authors
- [YOUR NAME](LINK OF YOUR GITHUB PAGE)  
    - main contributor  
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
    - correspondence  

## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- YOUR ADDRESS  
- tadahaya[at]gmail.com  
    - lead contact  