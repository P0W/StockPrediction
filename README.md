# Buy and sell predictions for stock

Abstract
---------
>The prediction of the share market value is of great importance to help in maximizing the profit of stock purchase while keeping the risk low. With latest advancement in technologies, the opportunity to gain a steady fortune from the share market is increased, which also helps experts to find out the most informative indicators to make a better prediction. 

>Machine learning has many applications, one of which is to forecast time series. One of the most interesting (or perhaps most profitable) time series to predict are, arguably, stock prices. 
Using features like the latest announcements about an organization, their quarterly revenue results, etc., machine learning techniques have the potential to unearth patterns and insights, that we did not see before, and these can be used to make unerringly accurate predictions.

>The analysis of historical stock data sets and extracting certain trends would help to predict the future value of the stock. For prediction, a recurrent neural network will be fed with a preprocessed historical value of a stock value for getting trained on the time series. Once trained the neural network later would be used for making some predictions for next trading day(s). Based on these predictions a potential Buys and Sells for given stock can be generated for a potential swing trade.

Stock Market - Buys and Sells
-----------------------------
>Companies need money to undertake projects. One of many ways to raise money is by issuing ownership in the company to public by means of issues share. Owning a share is akin to holding a portion of the company. These shares are then traded in share market, which these days now happens online only.

![The Rumours](https://github.com/P0W/StockPrediction/blob/master/build/img/Rumours.png)

>Since there is a trade involve, i.e. people purchase share from someone who is selling the same quantity of share, there’s always a place to “bargain”! Like general tendency for purchasing any physical quantity, people almost tend to buy at lower price and sell at higher price.


Overall Idea:
-------------
![Overall !dea](https://github.com/P0W/StockPrediction/blob/master/build/img/SwingTrade.png)

Commandline interface to train stocks:
--------------------------------------
![Commandline interface to train stocks](https://github.com/P0W/StockPrediction/blob/master/build/img/Cmdline.png)

Front-end interface to visualize trained data:
----------------------------------------------
![Front-end interface to visualize trained data](https://github.com/P0W/StockPrediction/blob/master/build/img/Frontend.png)

Sample Visulization using d3.js:
--------------------------------
![Sample Visualization](https://github.com/P0W/StockPrediction/blob/master/build/img/Prediction.png)

Dataset Preparation:
--------------------
> For feeding the data to the neural network, the stock data needs to be reshaped. For each 5 previous stock prices we would train for the 6th Stock Price!

![Dataset](https://github.com/P0W/StockPrediction/blob/master/build/img/Dataset.png)


The One Epoch:
--------------
![The One Epoch](https://github.com/P0W/StockPrediction/blob/master/build/img/TheEpoch.PNG)

Libraries Used:
---------------
![Libraries](https://github.com/P0W/StockPrediction/blob/master/build/img/Libraries.PNG)

The Design:
-----------
![Design](https://github.com/P0W/StockPrediction/blob/master/build/img/uml.png)


>----------------------------------------------------------------------------------------
How to setup:
-------------
1. Clone the repository

2. Make sure following are installed:
  
   a. cmake
   
   b. [Pytorch](https://pytorch.org/get-started/locally/)
   
   c. libcurl (for linux only, usually comes installed)
   
   d. [C++ Boost/Beast](https://github.com/boostorg/beast) Inspired by [this example](https://www.boost.org/doc/libs/develop/libs/beast/example/http/server/async/http_server_async.cpp). Requires date_time and regex boost library.
   
   d. Recommended:
      * Sign up [quandl.com](https://www.quandl.com/) for free stock prices dataset from Bombay Stock Exchanges. It provides REST service to download historical stock prices.
  
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


_BITS Final Semester Project_
**NOTE TO ALL MY DEAR BITS COLLEAGUES, IN CASE YOU ARE THINKING TO COPY THE WORK, PLEASE DONT DO SO AS BITS PLAGIARISM CHECKER TOOL IS INCREDIBLE !! YOU WILL BE AWARDED ZERO !  YOU HAVE BEEN WARNED !** 
