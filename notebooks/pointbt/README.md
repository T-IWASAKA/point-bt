# CimCLR-RBC
Deep learning model for red blood cell image feature extraction  

# ！最初にやること！
- READMEの変更  
    - repository nameの変更  
    - Authorsの追記  
    - Contactの追記  
- module内の変更  
    - 直下の__init__.pyのversionを適宜変更  
    - setup.py冒頭のdescriptionを更新  
    - setup.py冒頭のpython versionを確認  

# Note
This repository is under construction and will be officially released by [Mizuno group](https://github.com/mizuno-group).  
Please contact tadahaya[at]gmail.com before publishing your paper using the contents of this repository.  

## Install
- ``` pip install git+{URL OF THIS GITHUB REPOSITORY}@{BRANCH NAME} ```  
- In the development stage, it may be helpful to add ``` --force-reinstall ``` option  

## Organization
------------  

    ├── LICENSE  
    │
    ├── README.md               <- The top-level README for developers using this project.  
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, 
    │                              e.g. generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                <- Makes project pip installable (pip install -e .) so src can be imported.
    │                              Note that entry points should be modified for CLI
    │                              otherwise the latest CLI file will be employed.
    │
    └── module                  <- Source code for use in this project.
        │
        ├── core.py             <- Interactive runner of this module.
        │
        ├── cli.py              <- CLI runner of this module.
        │  
        └── src                 <- Main src for this module.

------------

## Authors
- [Takumi Iwasaka](https://github.com/T-IWASAKA/)  
    - main contributor  
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
    - correspondence  

## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:  
- takubasketrunner[at]gmail.com  
- tadahaya[at]gmail.com  
    - lead contact  