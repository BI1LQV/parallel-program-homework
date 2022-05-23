# Gemm Samples On CUDA

*If there is any bugs you found.*

*give me a pull request though git*

*https://github.com/DevilInChina/GemmSamples *~~damn github~~*

*or*

*https://gitee.com/devilinchina/gemm-samples* *recommended*



## Set Environment

* Check your GPU version

  ```shell
  nvidia-smi
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 450.119.03   Driver Version: 450.119.03   CUDA Version: 11.0     |
  |-------------------------------+----------------------+----------------------+
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |                               |                      |               MIG M. |
  |===============================+======================+======================|
  |   0  GeForce MX150       Off  | 00000000:01:00.0 Off |                  N/A |
  | N/A   60C    P0    N/A /  N/A |    608MiB /  2002MiB |      4%      Default |
  |                               |                      |                  N/A |
  +-------------------------------+----------------------+----------------------+
                                                                                 
  +-----------------------------------------------------------------------------+
  | Processes:                                                                  |
  |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
  |        ID   ID                                                   Usage      |
  |=============================================================================|
  |    0   N/A  N/A       951      G   /usr/lib/xorg/Xorg                 45MiB |
  |    0   N/A  N/A      1511      G   /usr/lib/xorg/Xorg                169MiB |
  |    0   N/A  N/A      1689      G   /usr/bin/gnome-shell              149MiB |
  |    0   N/A  N/A     17148      G   ...AAAAAAAAA= --shared-files      203MiB |
  |    0   N/A  N/A     18953      G   ...AAAAAAAAA= --shared-files       31MiB |
  +-----------------------------------------------------------------------------+
  ```

* if you got problem with this command, Use other way to install a nvidia-driver

  https://www.nvidia.com/Download/index.aspx

* As the code was write and compile at cuda-10.1, I suggest you to download cuda 10.1 too.

  https://developer.nvidia.com/cuda-toolkit-archive

* Flow the instruction and get install well done. nvcc can be write well.

  ```shell
  nvcc
  ```

  use
  
  ```shell
  nvcc -version
  ```
  
  to check version, get result like
  
  ```shell
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2019 NVIDIA Corporation
  Built on Sun_Jul_28_19:07:16_PDT_2019
  Cuda compilation tools, release 10.1, V10.1.243
  ```
  
  

## Compile

## Before

* **No matter using make or cmake or type command in**.

  **Be aware of nvcc option $-gencode=arch=compute\_61,code=compute\_61$**

  Different device with different number, check on https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/  or some place else. 

* **gcc version**

  In my laptop gcc later then 8 will not work, check before you compile.

### Use make to compile

* make project is prepare for users using vscode, or something IDE.
* Before you run commands below, I suggest you to check Makefile and set openBLAS path to correct.

```shell
cd makeVer
make
./gemm_test 100 100 100
```

### Use cmake to compile

* cmake project prepare for guys using clion, this CMakeLists.txt works well on Linux, perhaps you need extra works to make setting works well. 

```shell
cd cmakeVer
mkdir my_build;cd my_build
cmake ..
make
./gemm_test 100 100 100
```

