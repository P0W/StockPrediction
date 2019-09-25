/*
 * NetworkTrainer.cpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#include "NetworkTrainer.hpp"
#include <iostream>
#include "NetworkConstants.hpp"
#include "Timer.hpp"

#include <fstream>


namespace {
    void logFullyTrainedModel(const std::string& modelName, double lossVal, int64_t epoch, double time) {
        const std::string fileName = NetworkConstants::kRootFolder + "fullTrained.csv";
        std::ifstream checkHandle(fileName);
        bool fileExits = checkHandle.good();
        checkHandle.close();
        std::ofstream fileHandle(fileName, std::ios::app);
        if (fileHandle.good()) {
            if (!fileExits) {
                fileHandle << "Symbol,Loss,Epochs,Duration\n";
            }
            fileHandle << modelName << "," << lossVal << "," << epoch << "," << time << '\n';
        }
        fileHandle.close();
    }
}

NetworkTrainer::NetworkTrainer(int64_t input, int64_t hidden, int64_t output,
                               int64_t numLayers, int64_t prevSamples,
                               double learningRate, int64_t maxEpochs,
                               const std::string& modelName)
    :

      gpuAvailable(torch::cuda::is_available()),
      maxEpochs(maxEpochs),
      prevSamples(prevSamples),
      modelName(modelName),
      lstmNetwork(nullptr),
      optimizer(nullptr)

{
  torch::nn::LSTMOptions lstmOpts1(input, hidden);
  torch::nn::LSTMOptions lstmOpts2(hidden, hidden);
  lstmOpts1.layers(numLayers)
      .dropout(NetworkConstants::klsmt1DropOut)
      .with_bias(NetworkConstants::kIncludeBias);
  lstmOpts2.layers(numLayers)
      .dropout(NetworkConstants::klsmt2DropOut)
      .with_bias(NetworkConstants::kIncludeBias);

  torch::nn::LinearOptions linearOpts(hidden, output);
  linearOpts.with_bias(false);
  lstmNetwork = std::make_shared<StockLSTM>(lstmOpts1, lstmOpts2, linearOpts);

  torch::optim::AdamOptions opts(learningRate);
  optimizer = std::make_shared<torch::optim::Adam>(
      torch::optim::Adam(lstmNetwork->parameters(), opts));

  if (gpuAvailable) {
    std::cout << "Using CUDA\n";
    lstmNetwork->to(torch::kCUDA);
  }
}

NetworkTrainer::~NetworkTrainer() {}

torch::Tensor NetworkTrainer::fit(const torch::Tensor& x_train,
                                  const torch::Tensor& y_train) {
  torch::Tensor y_pred, input, target;
  torch::Tensor loss;
  input = x_train.view({prevSamples, -1, 1});


  if (gpuAvailable) {
    input = input.to(torch::kCUDA);
    target = y_train.to(torch::kCUDA);
  }
  else {
    target = y_train;
  }

  float running_loss = 1;
  int64_t epoch = 0;
  input.set_requires_grad(true);

  Timer t1("Total Elapsed Time: %2.f\n");
  Timer t2("Epoch Time: %.2f ");

  const std::string neuralNetLogFile = modelName + ".pt";
  const std::string predictLogFile = modelName + "_pred.csv";

  if (!modelName.empty()) {
    try {
      loadModel(neuralNetLogFile);
      std::cout << "Loaded a existing trained model\n";
    } catch (...) {
      std::cout << "Starting a fresh training...\n";
    }
  }

  while (running_loss > kRunningLoss) {
    // Zero out the gradients
    optimizer->zero_grad();

    // Predict the output using the neural network
    y_pred = lstmNetwork->forward(input);

    // Calculate loss using mean square error function
    loss = torch::mse_loss(y_pred, target);

    // Backward propagation
    loss.backward();

    // Step the optimizer
    optimizer->step();

    running_loss = loss.item<float>();
    if (epoch >= this->maxEpochs || t2 > NetworkConstants::kMaxTrainTime) {
      std::cout << "Loss is too high after epoch " << epoch << ": "
                << running_loss << std::endl;
      dataWriter(predictLogFile, y_pred);
      return y_pred;
    }

    else if (running_loss < NetworkConstants::kMinimumLoss) {
      std::cout << "Network fully trained!\n";
      logFullyTrainedModel(modelName, running_loss, epoch, t2);
      dataWriter(predictLogFile, y_pred);
      return y_pred;
    }

    else if (epoch % 10 == 0) {
      t2.show(false);
      std::cout << " epoch " << epoch << " [Running Loss = " << running_loss
                << " ( " << neuralNetLogFile << " ) ]\n";
      saveModel(neuralNetLogFile);
      dataWriter(predictLogFile, y_pred);
    }
    epoch++;
  }

  return y_pred;
}

void NetworkTrainer::saveModel(const std::string& fileName) const {
  torch::save(lstmNetwork, fileName);
}

void NetworkTrainer::loadModel(const std::string& fileName) {
  torch::load(lstmNetwork, fileName);
}
