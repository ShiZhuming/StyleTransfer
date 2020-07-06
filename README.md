# Image Style Transfer

## Introduction

This is a real-time image style transfer project based on AdaIn and VGG network , python flask as frontend and HTML5 as backend , project of AI Introduction, a course of Peking Unicersity

Our website is [http://pkuszm.cn/](http://pkuszm.cn/)

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

Use `start.py` to start web application
```
$ python3 start.py
```

It is recommended to use `virtualenv` and `gunicorn` to deploy the website on the server
```
$ pip install virtualenv
$ virtualenv -p /usr/bin/python3 ENVE
$ gunicorn -c gunicorn.conf.py --error-logfile errorlog.txt start:app -D
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
├── thumb.py
├── favicon.ico
├── image
│   ├── upload
│   │   ├── content
│   │   ├── style
│   │   └── convert
│   └── thumb
│       ├── content
│       ├── style
│       └── convert
├── static
├── templates
│   ├── index.html
│   ├── show.html
│   ├── authors.html
│   └── submissions.html
├── test.py
├── .gitignore
└── requirements.txt
```

## References

1. Gatys, Leon A., A. S. Ecker, and M. Bethge. "Image Style Transfer Using Convolutional
Neural Networks." Computer Vision & Pattern Recognition 2016.

2. Huang, Xun, and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance
Normalization." 2017.
