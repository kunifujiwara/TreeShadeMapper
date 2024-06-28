## Install required software

Please run the following pip and conda install commands:

# base

```
conda create --name canopyshade
#conda create --name myenv python=3.10
conda activate canopyshade
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
#conda install -c anaconda ipykernel
#python -m ipykernel install --user --name=canopyshade
```

# modules

```
pip install pandas
pip install zensvi
#pip install matplotlib
#pip install scikit-learn
pip install -U "ray[data,train,tune,serve]"
```
