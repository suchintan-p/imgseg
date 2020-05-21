# Realtime Image Segmenter

[![Build Status](https://travis-ci.org/pillarpond/image-segmenter-android.svg?branch=master)](https://travis-ci.org/pillarpond/image-segmenter-android)

This sample demonstrates realtime image segmentation on Android. The project is based on the [Deeplab](http://liangchiehchen.com/projects/DeepLab.html)

## Model
Tensorflow provide deeplab models pretrained several datasets. In this project, I used [mobilenetv2_coco_voc_trainaug](http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz)

## Inspiration
The project is heavily inspired by
* [Deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
* [DeepLab on Android](https://github.com/dailystudio/ml/tree/master/deeplab)
* [Tensorflow Android Camera Demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)

## Screenshots
![demo](./demo.gif)

## Pre-trained model
[DeepLab segmentation](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) (257x257) [&#91;download&#93;](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)

## License
[Apache License 2.0](./LICENSE)
