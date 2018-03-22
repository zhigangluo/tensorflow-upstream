<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>


## Tensorflow ROCm port

This project is based on TensorFlow 1.3.0. It has been verified to work with the latest ROCm1.7.1 release. Please follow the instructions [here](https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/quick-start.md) to set up your ROCm stack.

A docker container: **rocm/tensorflow:rocm1.7.1(https://hub.docker.com/r/rocm/tensorflow/)** is readily available to be used.

The Wheels:
- Python 2: http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.3.0-cp27-cp27mu-linux_x86_64.whl
- Python 3: http://repo.radeon.com/rocm/misc/tensorflow/tensorflow-1.3.0-cp35-cp35m-linux_x86_64.whl

For details on Tensorflow ROCm port, please take a look at the [ROCm-specific README file](README.ROCm.md).

-----------------

| **`Linux CPU`** | **`Linux GPU`** | **`Mac OS CPU`** | **`Windows CPU`** | **`Android`** |
|-----------------|---------------------|------------------|-------------------|---------------|
| [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-cpu)](https://ci.tensorflow.org/job/tensorflow-master-cpu) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-linux-gpu)](https://ci.tensorflow.org/job/tensorflow-master-linux-gpu) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-mac)](https://ci.tensorflow.org/job/tensorflow-master-mac) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-win-cmake-py)](https://ci.tensorflow.org/job/tensorflow-master-win-cmake-py) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-android)](https://ci.tensorflow.org/job/tensorflow-master-android) |

**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow also includes TensorBoard, a data visualization toolkit.

TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence Research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

**If you'd like to contribute to TensorFlow, be sure to review the [contribution
guidelines](CONTRIBUTING.md).**

**We use [GitHub issues](https://github.com/tensorflow/tensorflow/issues) for
tracking requests and bugs, but please see
[Community](https://www.tensorflow.org/community/) for general questions
and discussion.**

## For more information

* [TensorFlow website](https://www.tensorflow.org)
* [TensorFlow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
* [TensorFlow Model Zoo](https://github.com/tensorflow/models)
* [TensorFlow MOOC on Udacity](https://www.udacity.com/course/deep-learning--ud730)

The TensorFlow community has created amazing things with TensorFlow, please see the [resources section of tensorflow.org](https://www.tensorflow.org/about/#community) for an incomplete list.
