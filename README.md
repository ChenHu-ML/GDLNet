# README
This repository is the official code for our IJCAI 2024 paper: "A Grassmannian Manifold Self-Attention Network for Signal Classification". [GDLNet](https://www.ijcai.org/proceedings/2024/0564.pdf)

If you have any problems, please don't hesitate to contact me.

## Requirements

To install necessary dependencies by conda, run the following command:
```setup
conda env create -f GDLNet.yml
conda activate GDLNet
```

## Dataset
Please download the datasets and put them in the folder 'data'.

1. MAMEM-SSVEP-II:
   https://www.mamem.eu/results/datasets/
2. BCI-ERN:
    https://www.kaggle.com/competitions/inria-bci-challenge/data

Link to download [data](https://drive.google.com/file/d/1RxN2PWOkYJw-NzyM0vaxdLkina2q-_Rj/view?usp=sharing)

## Training and testing

To train and test the experiments on the Mamem and Bcicha datasets, run this command:

```train and test
python GDLNet_mamem.py
python GDLNet_baicha.py
```

## Reference
```bash
@inproceedings{pan2022matt,
  title={MAtt: a manifold attention network for EEG decoding},
  author={Pan, Yue-Ting and Chou, Jing-Lun and Wei, Chun-Shu},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={31116--31129},
  year={2022}
}
```

```bash
@inproceedings{wang2024grassmannian,
  title={A Grassmannian Manifold Self-Attention Network for Signal Classification},
  author={Wang, Rui and Hu, Chen and Chen, Ziheng and Wu, Xiao-Jun and Song, Xiaoning},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages={5099--5107},
  year={2024}
}

```

