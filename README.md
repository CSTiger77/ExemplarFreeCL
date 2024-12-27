Environment contruction:

Anaconda:

wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

chmod +x Anaconda3-2024.06-1-Linux-x86_64.sh

./Anaconda3-2024.06-1-Linux-x86_64.sh

conda create -n kcli python=3.9

pytorch:

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install yacs -i https://pypi.tuna.tsinghua.edu.cn/simple

ERA_DFCL_main.py : ERA-EFCL

foster_main.py: FOSTER-EFCL

ABD_main.py: ABD

RDFCL_main.py: RDFCIL

.
.
.


Run ERA-EFCL's Experiments: python ERA_DFCL_main.py 

Run FOSTER-EFCL's Experiments: python foster_main.py

others: python ABD_main.py; python RDFCL_main.py, ...

Acknowledge: Code is based on the sourde codes of 

FOSTER (https://github.com/G-U-N/ECCV22-FOSTER), 

ABD (https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL), 

R-DFCIL (https://github.com/jianzhangcs/R-DFCIL), 

FeTrIL (https://github.com/GregoirePetit/FeTrIL.), 

FeCAM (https://github.com/dipamgoswami/FeCAM),

NAPAVQ (https://github.com/TamashaM/NAPA-VQ.git).
