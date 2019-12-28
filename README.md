# StereoZED
Trained and traced StereoNet network with ZED Camera streaming input.

## Prerequisites:

1. nVidia driver version >= 440.3301
2. CUDA 10.0
3. ZED API
4. OpenCV
5. libtorch

## Download:

To download the source code, simply clone the repo to your home folder:

```console
foo@bar:~$ cd ~
foo@bar:~$ git clone https://github.com/abexultan/StereoZED.git
Cloning into 'StereoZED'...
remote: Enumerating objects: 30, done.
remote: Counting objects: 100% (30/30), done.
remote: Compressing objects: 100% (29/29), done.
remote: Total 30 (delta 12), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (30/30), done.
Checking connectivity... done.
foo@bar:~$ ls StereoZED/
CMakeLists.txt  README.md  src  stereonet_traced_720p.pt
```

## Build:

```console
foo@bar:~/StereoZED$ mkdir build && cd build
foo@bar:~/StereoZED/build$ cmake -DCMAKE_PREFIX_PATH=/<path-to-libtorch> ..
-- The C compiler identification is GNU 5.4.0
-- The CXX compiler identification is GNU 5.4.0
-- Check for working C compiler: /usr/bin/cc
-- Check for working C compiler: /usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/c++
-- Check for working CXX compiler: /usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for sgemm_
-- Looking for sgemm_ - found
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- A library with BLAS API found.
-- Found CUDA: /usr/local/cuda-10.0 (found suitable version "10.0", minimum required is "10") 
-- Found CUDA: /usr/local/cuda-10.0 (found suitable version "10.0", minimum required is "10.0") 
-- Found CUDA: /usr/local/cuda-10.0 (found version "10.0") 
-- Caffe2: CUDA detected: 10.0
-- Caffe2: CUDA nvcc is: /usr/local/cuda-10.0/bin/nvcc
-- Caffe2: CUDA toolkit directory: /usr/local/cuda-10.0
-- Caffe2: Header version is: 10.0
-- Found CUDNN: /home/kist/cuda/lib64/libcudnn.so  
-- Found cuDNN: v7.6.5  (include: /home/kist/cuda/include, library: /home/kist/cuda/lib64/libcudnn.so)
-- Autodetected CUDA architecture(s):  6.1 6.1
-- Added CUDA NVCC flags for: -gencode;arch=compute_61,code=sm_61
-- Found torch: <path-to-libtorch.so> 
-- Configuring done
-- Generating done
-- Build files have been written to: /home/foo/StereoZED/build
foo@bar:~/StereoZED/build$ make
[ 50%] Building CXX object CMakeFiles/zed_stereonet.dir/src/zed_stereonet.cpp.o
[100%] Linking CXX executable zed_stereonet
[100%] Built target zed_stereonet
```

## Use:
```console
foo@bar:~/StereoZED/build$ ./zed_stereonet ../stereonet_traced_720p.pt 
```
Press **Q** to terminate the process.
