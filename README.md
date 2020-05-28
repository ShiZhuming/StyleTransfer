# Image Style Transfer

## Introduction

This is a project of AI Introduction , a course of Peking University.

Our website is [http://pkuszm.cn/show](http://pkuszm.cn)

## Authors

Zhuming Shi, Weibo Xu, Yiming Zhao

## Dependence

`Python 3` only
```
$ apt-get install python3.6
```

Use `requirements.txt` to install packages
```
$ pip3 install -r requirements.txt
```

## Structure

```
StyleTransfer/
├── LICENSE
├── README.md
├── start.py
├── function.py
├── net.py
├── train.py
├── convert.py
├── favicon.ico
├── image
|   └──  upload
|       ├── content
|       ├── style
|       └── convert
├── static
├── templates
|   ├── index.html
|   ├── show.html
|   └── submissions.html
├── test.py
├── .gitignore
└── requirements.txt
```

## References

1. Gatys, Leon A., A. S. Ecker, and M. Bethge. "Image Style Transfer Using Convolutional
Neural Networks." Computer Vision & Pattern Recognition 2016.

2. Huang, Xun, and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance
Normalization." 2017.
