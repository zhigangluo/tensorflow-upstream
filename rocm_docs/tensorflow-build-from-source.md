# TensorFlow ROCm port: Building From Source

## Intro
This instruction provides a starting point for build TensorFlow ROCm port from source.
*Note*: it is recommended to start with a clean Ubuntu 16.04 system

## Install hcFFT 
```
git clone https://github.com/ROCmSoftwarePlatform/hcFFT.git ~/hcfft
cd ~/hcfft && bash build.sh && sudo dpkg -i ~/hcfft/build/*.deb
```

## Install required python packages
```
sudo apt-get update && sudo apt-get install -y \
    python-numpy \
    python-dev \
    python-wheel \
    python-mock \
    python-future \
    python-pip \
    python-yaml \
    python-setuptools && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*
```

## Install bazel
```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y openjdk-8-jdk openjdk-8-jre unzip && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/* 
cd ~/ && wget https://github.com/bazelbuild/bazel/releases/download/0.4.5/bazel-0.4.5-installer-linux-x86_64.sh 
sudo bash ~/bazel*.sh
```

## Build TensorFlow ROCm port
```
# Clone it
cd ~ && git clone https://github.com/ROCmSoftwarePlatform/tensorflow.git

# Configure TensorFlow ROCm port
# Enter all the way
cd ~/tensorflow && ./configure 

# Build and install TensorFlow ROCm port pip package
./build
```

## Clone TensorFlow models and benchmarks
```
cd ~ && git clone https://github.com/soumith/convnet-benchmarks.git
cd ~ && git clone https://github.com/tensorflow/models.git
```
