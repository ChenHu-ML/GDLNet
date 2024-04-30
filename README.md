# README
This repository is the official code for our IJCAI 2024 paper: "A Grassmannian Manifold Self-Attention Network for Signal Classification". [GDLNet]()

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

Link to download [data](https://drive.google.com/file/d/1_KBfSNzvxCZ-HwiOASQhlFe8wwsq4vHt/view?usp=sharing)

## Training and testing

To train and test the experiments on the Mamem and Bcicha datasets, run this command:

```train and test
python GDLNet_mamem.py
python GDLNet_baicha.py
```

## Reference
```bash
@article{pan2022matt,
  title={MAtt: a manifold attention network for EEG decoding},
  author={Pan, Yue-Ting and Chou, Jing-Lun and Wei, Chun-Shu},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={31116--31129},
  year={2022}
}
```


```

