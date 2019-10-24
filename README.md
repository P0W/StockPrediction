# StockPrediction
_BITS Final Semester Project_

1. Clone the repository

2. Make sure following are installed:
  
   a. cmake
   
   b. [Pytorch](https://pytorch.org/get-started/locally/)
   
   c. libcurl (for linux only, usually comes installed)
   
   d. [C++ Boost/Beast](https://github.com/boostorg/beast) Inspired by [this example](https://www.boost.org/doc/libs/develop/libs/beast/example/http/server/async/http_server_async.cpp). Requires date_time and regex boost library.
   
   d. Recommended:
  
      NVIDIA CUDA 10.1 install for leveraging GPU acceleration if the graphics card suports it
      * Check if supported [here](https://developer.nvidia.com/cuda-gpus)
      
      * Install CUDA Toolkit from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork)
      
      * Install Deep Learning SDK cuDNN from [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
      
      * On Windows, GPU support was only available when pytorch was built from source
         ..* Recursive clone pytorch library
         
         * `git clone --recursive https://github.com/pytorch/pytorch`
         
         * Launch VS Developer Command Prompt
         
         * CD to clones pytorch folder.
         
         * `python setup.py install`

3. For Windows, need Microsoft Visual Studio >=17

4. Build the CMAKE project

   a. On Windows:
      
      `C:\Program Files\CMake\bin\cmake.exe" -DCMAKE_PREFIX_PATH=$PWD/../../pytorch/torch  cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..`
      
      * On generated project add environment path
      
      `PATH=%PATH%;D:\playground\pytorch\torch\lib;C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64`
      
      * Manually fix the libraries:
      
         * urlmon.lib
         * D:\playground\pytorch\torch\lib\c10.lib
         * C:\Program Files\NVIDIA Corporation\NvToolsExt\lib\x64\nvToolsExt64_1.lib
         * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cudart_static.lib
         * D:\playground\pytorch\torch\lib\caffe2_nvrtc.lib
         * D:\playground\pytorch\torch\lib\c10_cuda.lib
         * D:\playground\pytorch\torch\lib\torch.lib
         * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cufft.lib
         * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\curand.lib
         * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cudnn.lib
         * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\cublas.lib

      
   b. On Linux:
   
      `cmake -DCMAKE_PREFIX_PATH=$PWD/../libtorch ..`
