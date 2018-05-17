# Keras Benchmarks

# Table of Contents

1. [Overview](#overview)
2. [Library Versions](#library-versions)
3. [CNN Benchmarks](#cnn-benchmarks)
    1. [CNN Benchmark Results](#cnn-benchmark-results)
4. [RNN Benchmarks](#rnn-benchmarks)
    1. [RNN Benchmark Results](#rnn-benchmark-results)
5. [Setup](#setup)
6. [How to Run CNN Benchmarks](#how-to-run-cnn-benchmarks)
7. [How to Run RNN Benchmarks](#how-to-run-rnn-benchmarks)
8. [References](#references)

## Overview

This Benchmark module provides an easy to use scripts for benchmarking various Convolutional Neural Network (CNN) and 
Recurrent Neural Network (RNN) Keras models. You can use these scripts to generate benchmarks on a CPU, one GPU or 
multi-GPU instances. Apache MXNet and TensorFlow backends are supported for generating the benchmark results.


`CREDITS:` This benchmark module borrows and extends the benchmark utility from [TensorFlow Keras benchmarks]
(https://github.com/tensorflow/benchmarks/tree/keras-benchmarks).

## Library Versions

```
NOTE:
    The below benchmarks use native pip packages provided by the frameworks without any optimized compile builds. See the setup section for more details.

```
| Framework | Version | Installation |
| --- | --- | --- |
|  Keras | 2.1.6 | pip install keras-mxnet  |
|  MXNet (CPU) | 1.2 | pip install mxnet-mkl  |
| MXNet (GPU) | 1.2 | pip install mxnet-cu90 |
| TensorFlow (CPU) | 1.8 | pip install tensorflow |
| TensorFlow (GPU) | 1.8 | pip install tensorflow-gpu |
| CUDA | 9.0 | |
| cuDNN | 7.0.5 | |
 
## CNN Benchmarks

Currently, this utility helps in benchmarking the following CNN networks:
1. [ResNet56](https://arxiv.org/abs/1512.03385)

Currently, this utility helps in benchmarking on the following datasets:
1. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
2. [ImageNet](http://image-net.org/download)
3. Synthetic data

```
NOTE:
    1. For CIFAR10 and synthetic data, the benchmark scripts will download/generate the required data.
    2. For ImageNet data, you are expected to download the data - http://image-net.org/download
    3. You can benchmark with a different number of layers in ResNet.

```

### CNN Benchmark Results

#### ResNet50-ImageNet

| Instance Type | GPUs  | Batch Size  | Keras-MXNet (img/sec)  | Keras-TensorFlow (img/sec)  |
|---|---|---|---|---|
|  P3.8X Large | 1  | 32  | 165  | 50  |
|  P3.8X Large |  4 |  128 | 538  | 162  |
|  P3.16X Large | 8  | 256  | 728  | 212  |

#### ResNet50-Synthetic Data

| Instance Type | GPUs  | Batch Size  | Keras-MXNet (img/sec)  | Keras-TensorFlow (img/sec)  |
|---|---|---|---|---|
|  C5.18X Large | 0  | 32  | 9  | 4  |
|  P3.8X Large |  1 |  32 | 229  | 164  |
|  P3.8X Large |  4 |  128 | 728  | 409  |
|  P3.16X Large | 8  | 256  | 963  | 164  |


#### ResNet50-CIFAR10

| Instance Type | GPUs  | Batch Size  | Keras-MXNet (img/sec)  | Keras-TensorFlow (img/sec)  |
|---|---|---|---|---|
|  C5.18X Large | 0  | 32  | 87  | 59  |
|  P3.8X Large | 1  | 32  | TBD  | TBD  |
|  P3.8X Large |  4 |  128 | 1792  | 1020  |
|  P3.16X Large | 8  | 256  | 1618  | 962  |


You can see more benchmark experiments with different instance types, batch_size and other parameters in [detailed CNN 
results document](benchmark_result/CNN_result.md).

 
```
NOTE:
    1. Image_data_format for MXNet backend - 'channels_first'
    2. Image_data_format for TensorFlow backend - 'channels_last'
    3. C5 instance details - https://aws.amazon.com/ec2/instance-types/c5/
    4. P3 instance details (Volta GPU) - https://aws.amazon.com/ec2/instance-types/p3/
```

## RNN Benchmarks

Currently, this utility helps in benchmarking the following RNN networks:
1. [LSTM Text Generation](https://github.com/awslabs/keras-apache-mxnet/blob/master/benchmark/scripts/models/lstm_text_generation.pyy)

Currently, this utility helps in benchmarking on the following datasets:
1. [Nietzsche](https://s3.amazonaws.com/text-datasets/nietzsche.txt)
2. [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset)
3. Synthetic data

### RNN Benchmark Results

#### LSTM-Nietzsche

| Instance Type | GPUs  | Batch Size  | Keras-MXNet (Time/Epoch), (GPU Mem)   | Keras-TensorFlow (Time/Epoch), (GPU Mem)   |
|---|---|---|---|---|
|  C5.18X Large | 0  | 128  | 78 sec, N/A | 55 sec, N/A|
|  P3.8X Large |  1 |  128 | 52 sec, 792 MB | 51 sec, 15360 MB|
|  P3.8X Large | 4  | 128  | 47 sec, 770 MB | 87 sec, 15410 MB |
|  P3.16X Large | 8  | 128  | TBD | TBD |

#### LSTM-WikiText2

| Instance Type | GPUs  | Batch Size  | Keras-MXNet (Time/Epoch), (GPU Mem)  | Keras-TensorFlow (Time/Epoch), (GPU Mem)  |
|---|---|---|---|---|
|  C5.18X Large | 0  | 128  | 1345 sec, N/A  | 875, N/A  |
|  P3.8X Large |  1 |  128 | 868 sec, 772 MB | 817, 15360 MB  |
|  P3.8X Large | 4  | 128  | 775 sec, 764 MB | 1468, 15410 MB  |
|  P3.16X Large | 8  | 128  | TBD | TBD |

#### Synthetic Data

| Instance Type | GPUs  | Batch Size  | Keras-MXNet (Time/Epoch), (GPU Mem)   | Keras-TensorFlow (Time/Epoch), (GPU Mem)   |
|---|---|---|---|---|
|  C5.18X Large | 0  | 128  | 24 sec, N/A | 14 sec, N/A|
|  P3.8X Large |  1 |  128 | 13 sec, 792 MB | 12 sec, 15360 MB|
|  P3.8X Large | 4  | 128  | 12 sec, 770 MB | 21 sec, 15410 MB |
|  P3.16X Large | 8  | 128  | TBD | TBD |

You can see more benchmark experiments with different instance types, batch_size and other parameters in [detailed RNN 
results document](benchmark_result/RNN_result.md).

## Setup

1. Install Keras-MXNet following the [installation guide](../docs/mxnet_backend/installation.md). 
2. You need to install [Apache MXNet](http://mxnet.incubator.apache.org/install/index.html) and/or [TensorFlow](https://www.tensorflow.org/install/) 
for running the benchmarks on the respective backend. Install the correct version of the backend based on your instance type (CPU/GPU).
3. Download this [awslabs/keras-apache-mxnet](https://github.com/awslabs/keras-apache-mxnet) repository that contains
 all the benchmarking utilities and scripts.

```
    # Install Keras-MXNet
    $ pip install keras-mxnet
    
    # Install the backend - MXNet and/or TensorFlow
    # For MXNet 
    $ pip install mxnet-mkl # CPU
    $ pip install mxnet-cu90 # GPU
    
    # For TensorFlow
    $ pip install tensorflow # CPU
    $ pip install tensorflow-gpu # GPU
    
    # Download the source code for benchmarking
    $ git clone https://github.com/awslabs/keras-apache-mxnet
    $ cd keras-apache-mxnet/benchmark/scripts
```

## How to Run CNN Benchmarks


### ResNet50-ImageNet

Update the `~/.keras/keras.json` to set the `backend` and `image_data_format`.

For TensorFlow backend benchmarks, set `backend: tensorflow` and `image_data_format: channels_last`.
For MXNet backend benchmarks, set `backend: mxnet` and `image_data_format: channels_first`.

```
    $ python benchmark_resnet.py --dataset imagenet --version 1 --layers 56 --gpus 4 --epoch 20 --train_mode train_on_batch --data_path home/ubuntu/imagenet/train/

```
- version: can be 1 or 2 for ResNetv1 and ResNetv2 respectively.
- layers: Number of layers in ResNet
- gpus: Number of GPUs to be used. 0 to run on CPU
- train_mode: Since imagenet is a large dataset, you can choose 'train_on_batch' or 'fit_generator' to feed the data. We recommend 'train_on_batch'.
- data_path: Path where you have downloaded the ImageNet data.

### ResNet50-CIFAR10

Update the `~/.keras/keras.json` to set the `backend` and `image_data_format`.

For TensorFlow backend benchmarks, set `backend: tensorflow` and `image_data_format: channels_last`.
For MXNet backend benchmarks, set `backend: mxnet` and `image_data_format: channels_first`.

```
    $ python benchmark_resnet.py --dataset cifar10 --version 1 --layers 56 --gpus 4 --epoch 20
```
Set number of gpus, epochs based on your experiments.


### ResNet50-Synthetic

We have a utility shell script that you can run for benchmarking on the synthetic data.

For MXNet backend benchmarks:
```
    $ sh run_mxnet_backend.sh cpu_config resnet50 False 20 # For CPU Benchmarks
    $ sh run_mxnet_backend.sh gpu_config resnet50 False 20 # For 1 GPU Benchmarks
    $ sh run_mxnet_backend.sh 4_gpu_config resnet50 False 20 # For 4 GPU Benchmarks
    $ sh run_mxnet_backend.sh 8_gpu_config resnet50 False 20 # For 8 GPU Benchmarks
```

For TensorFlow backend benchmarks:
```
    $ sh run_tf_backend.sh cpu_config resnet50 False 20 # For CPU Benchmarks
    $ sh run_tf_backend.sh gpu_config resnet50 False 20 # For 1 GPU Benchmarks
    $ sh run_tf_backend.sh 4_gpu_config resnet50 False 20 # For 4 GPU Benchmarks
    $ sh run_tf_backend.sh 8_gpu_config resnet50 False 20 # For 8 GPU Benchmarks
```

The last parameter, 20, in the command is the number of epoch.

## How to Run RNN Benchmarks

#### LSTM-Nietzsche

You can use the utility shell script to run the RNN benchmark on the Nietzsche dataset.

For MXNet backend benchmarks:
```
    $ sh run_mxnet_backend.sh cpu_config lstm_nietzsche False 10 # For CPU Benchmarks
    $ sh run_mxnet_backend.sh gpu_config lstm_nietzsche False 10 # For 1 GPU Benchmarks
    $ sh run_mxnet_backend.sh 4_gpu_config lstm_nietzsche False 10 # For 4 GPU Benchmarks
    $ sh run_mxnet_backend.sh 8_gpu_config lstm_nietzsche False 10 # For 8 GPU Benchmarks
```

For TensorFlow backend benchmarks:
```
    $ sh run_tf_backend.sh cpu_config lstm_nietzsche False 20 # For CPU Benchmarks
    $ sh run_tf_backend.sh gpu_config lstm_nietzsche False 20 # For 1 GPU Benchmarks
    $ sh run_tf_backend.sh 4_gpu_config lstm_nietzsche False 20 # For 4 GPU Benchmarks
    $ sh run_tf_backend.sh 8_gpu_config lstm_nietzsche False 20 # For 8 GPU Benchmarks
```

#### LSTM-WikiText2

You can use the utility shell script to run the RNN benchmark on the WikiText2 dataset.

For MXNet backend benchmarks:
```
    $ sh run_mxnet_backend.sh cpu_config lstm_wikitext2 False 10 # For CPU Benchmarks
    $ sh run_mxnet_backend.sh gpu_config lstm_wikitext2 False 10 # For 1 GPU Benchmarks
    $ sh run_mxnet_backend.sh 4_gpu_config lstm_wikitext2 False 10 # For 4 GPU Benchmarks
    $ sh run_mxnet_backend.sh 8_gpu_config lstm_wikitext2 False 10 # For 8 GPU Benchmarks
```

For TensorFlow backend benchmarks:
```
    $ sh run_tf_backend.sh cpu_config lstm_wikitext2 False 20 # For CPU Benchmarks
    $ sh run_tf_backend.sh gpu_config lstm_wikitext2 False 20 # For 1 GPU Benchmarks
    $ sh run_tf_backend.sh 4_gpu_config lstm_wikitext2 False 20 # For 4 GPU Benchmarks
    $ sh run_tf_backend.sh 8_gpu_config lstm_wikitext2 False 20 # For 8 GPU Benchmarks
```


#### Synthetic Data

You can use the utility shell script to run the RNN benchmark on the Synthetic dataset.

For MXNet backend benchmarks:
```
    $ sh run_mxnet_backend.sh cpu_config lstm_synthetic False 10 # For CPU Benchmarks
    $ sh run_mxnet_backend.sh gpu_config lstm_synthetic False 10 # For 1 GPU Benchmarks
    $ sh run_mxnet_backend.sh 4_gpu_config lstm_synthetic False 10 # For 4 GPU Benchmarks
    $ sh run_mxnet_backend.sh 8_gpu_config lstm_synthetic False 10 # For 8 GPU Benchmarks
```

For TensorFlow backend benchmarks:
```
    $ sh run_tf_backend.sh cpu_config lstm_synthetic False 20 # For CPU Benchmarks
    $ sh run_tf_backend.sh gpu_config lstm_synthetic False 20 # For 1 GPU Benchmarks
    $ sh run_tf_backend.sh 4_gpu_config lstm_synthetic False 20 # For 4 GPU Benchmarks
    $ sh run_tf_backend.sh 8_gpu_config lstm_synthetic False 20 # For 8 GPU Benchmarks
```

## References

* [TensorFlow Keras Benchmarks](https://github.com/tensorflow/benchmarks/tree/keras-benchmarks/scripts/keras_benchmarks)
* [lstm_text_generation.py](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)
