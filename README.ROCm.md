# Tensorflow ROCm port #

## Introduction ##

This repository hosts the port of [Tensorflow](https://github.com/tensorflow/tensorflow) on ROCm platform. It uses various technologies on ROCm platform such as HIP and MIOpen. For details on HIP, please refer [here](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP). Optimized DNN library calls (via [MIOpen](https://github.com/ROCmSoftwarePlatform/MIOpen)) are also supported within this codebase.

## Installation ##

For further background information on ROCm, refer [here](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md).

The project is derived from TensorFlow 1.3.0 and has been verified to work with the latest ROCm 1.7.1 release.

For details on installation and usage, see these links:
* [Basic installation](rocm_docs/tensorflow-install-basic.md)
* [Building from source](rocm_docs/tensorflow-build-from-source.md)
* [Quickstart guide](rocm_docs/tensorflow-quickstart.md)
* [List of supported operators on ROCm](rocm_docs/core_kernels.md)


## Known Issues / Workarounds

### tensorflow/benchmarks workaround for TF v1.3
Since our current port of TF supports v1.3, we can't use some of the newest commits in `tensorflow/benchmarks`.  One specific error you could observe is a `ImportError: cannot import name batching`.  "Batching" was introduced in this [commit](https://github.com/tensorflow/benchmarks/commit/82dd0539c76afa8491e50d8f796e686b4d97b988).

We have to drop back to an earlier point in the commit history:  
```
git clone https://github.com/tensorflow/benchmarks.git 
git checkout -b oct23 f5d85aef2851881001130b28385795bc4c59fa38
```
