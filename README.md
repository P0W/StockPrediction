# StockPrediction
BITS Final Semester Project

1. Clone the repository
2. Make sure following are installed:
   a. cmake
   b. pytorch
   c. libcurl (for linux only)
   d. Recommended:
      NVIDIA CUDA 10.0 install for leveraging GPU acceleration if the graphics card suports it
      * Check if supported here: https://developer.nvidia.com/cuda-gpus
      * Install CUDA Toolkit from here: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork
3. For Windows, need Microsoft Visual Studio >=17
4. Got to build directory
   a. On Windows
      C:\Program Files\CMake\bin\cmake.exe" -DCMAKE_PREFIX_PATH=$PWD/../../libtorch  cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
   b. On Linux
      cmake -DCMAKE_PREFIX_PATH=$PWD/../libtorch ..
