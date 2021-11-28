# CMC_MoCo
Implementation of the algorithm Contrastive Multiview Coding (CMC) combined with Momentum Contrast (MoCo) for self-supervised learning of images.

## Setup

* Install conda environment
```shell
conda env create -f env.yml
```
* Download STL-10 dataset
```shell
python stl10_input.py
```

## Training

* Check configs in ``cfgs/config.yaml``
* Train the AlexNet encoder
```shell
python train
```
* Show metric evolutions in tensorboard
```shell
tensorboard --logdir exp_local
```

## Resources
* [Contrastive Multiview Coding (CMC) paper](https://arxiv.org/abs/1906.05849)
* [Momentum Contrast (MoCo) paper](https://arxiv.org/abs/1911.05722)
* [Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning](https://arxiv.org/abs/2107.09645)
