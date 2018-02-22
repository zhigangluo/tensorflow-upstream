# How to use on ROCm Machine

## Installation

```shell
$ git checkout rocm-v1

$ ./configure 
You have bazel 0.5.4 installed.
Please specify the location of python. [Default is /usr/bin/python]: 
Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]

Using python library path: /usr/local/lib/python2.7/dist-packages
Do you wish to build TensorFlow with MKL support? [y/N] n
No MKL support will be enabled for TensorFlow
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 
Do you wish to use jemalloc as the malloc implementation? [Y/n] n
jemalloc disabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] n
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N] n
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] y
XLA JIT support will be enabled for TensorFlow
Do you wish to build TensorFlow with VERBS support? [y/N] n
No VERBS support will be enabled for TensorFlow
Do you wish to build TensorFlow with OpenCL support? [y/N] n
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] n
No CUDA support will be enabled for TensorFlow
Do you wish to build TensorFlow with ROCm support? [y/N] y
CUDA support will be enabled for TensorFlow
Do you wish to build TensorFlow with MPI support? [y/N] n
MPI support will not be enabled for TensorFlow
Configuration finished

$ bazel build --config=opt --config=rocm //tensorflow/tools/pip_package:build_pip_package 

```

## Known Issues / Workarounds

### tensorflow/benchmarks workaround for TF v1.3
Since our current port of TF supports v1.3, we can't use some of the newest commits in `tensorflow/benchmarks`.  One specific error you could observe is a `ImportError: cannot import name batching`.  "Batching" was introduced in this [commit](https://github.com/tensorflow/benchmarks/commit/82dd0539c76afa8491e50d8f796e686b4d97b988).

We have to drop back to an earlier point in the commit history:  
```
git clone https://github.com/tensorflow/benchmarks.git 
git checkout -b oct23 f5d85aef2851881001130b28385795bc4c59fa38
```
