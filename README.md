# SSVDD_Python

## Subspace Support Vector Data Description model

A model which projects the positive class data into lower-dimensional feature space and makes hyper-spherical boundary around positive class while keeping the negative class data out of the boundary.

## Requirements

The SVDD model is based on Support Vector Data Description model and hence, for implementation purpose, we have used the openly available python implementation of [SVDD](https://github.com/iqiukp/SVDD-Python/blob/master/src/BaseSVDD.py). Therefore, add this to the directory before implementing SSVDD.

## Demo

The demo of SSVDD is given [HERE](https://github.com/Zaffarr/Credit_card_fraud_detection_using_SSVDD/tree/main).

## Citations

To use any part of this implementation, please cite the following papers.

```
@inproceedings{sohrab2018subspace,
  title={Subspace support vector data description},
  author={Sohrab, Fahad and Raitoharju, Jenni and Gabbouj, Moncef and Iosifidis, Alexandros},
  booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
  pages={722--727},
  year={2018},
  organization={IEEE}
}

@article{zaffar2023credit,
  title={Credit Card Fraud Detection with Subspace Learning-based One-Class Classification},
  author={Zaffar, Zaffar and Sohrab, Fahad and Kanniainen, Juho and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:2309.14880},
  year={2023}
}
```
