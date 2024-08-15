# DEDN
This repository provides the code for the paper [Dual Expert Distillation Network for Generalized Zero-Shot Learning](https://arxiv.org/pdf/2404.16348) (IJCAI 2024).

## Requirements
We conduct experiments on a RTX 4090.
```
- cuda=11.8
- python=3.8.18
- torch=2.0.0
- torchvision=0.15.1
```

## Data Preparation

1.Download datasets and put them in ‘./data'. We use three datasets [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/) following the data split of [xlsa17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip).

```
.
├── data
│   ├── CUB/CUB_200_2011/...
│   ├── SUN/images/...
│   ├── AWA2/Animals_with_Attributes2/...
│   └── xlsa17/data/...
└── ···
```

2.Run the script preprocessing.py to preprocess the data. For example:

```
python preprocessing.py --dataset CUB --compression --device cuda:0
python preprocessing.py --dataset SUN --compression --device cuda:0
python preprocessing.py --dataset AWA2 --compression --device cuda:0
```

## Train

Modify command in script.sh and run the script to train. 

```
sh script.sh
```

## Citation
If this work is helpful for you, please cite our paper.

```
@article{rao2024dual,
  title={Dual Expert Distillation Network for Generalized Zero-Shot Learning},
  author={Rao, Zhijie and Guo, Jingcai and Lu, Xiaocheng and Liang, Jingming and Zhang, Jie and Wang, Haozhao and Wei, Kang and Cao, Xiaofeng},
  journal={arXiv preprint arXiv:2404.16348},
  year={2024}
}
```
