/*
 * NetworkTrainer.cpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#include "NetworkTrainer.hpp"
#include "NetworkConstants.hpp"
#include "StockLSTM.hpp"
#include "Timer.hpp"
#include <fstream>
#include <iostream>

#include <torch/torch.h>

namespace {
void logFullyTrainedModel(const std::string &modelName,
                          const std::string &companyName, double lossVal,
                          int64_t epoch, double time) {
  const std::string fileName =
      NetworkConstants::kRootFolder + "fullTrained.csv";
  std::ifstream checkHandle(fileName);
  bool fileExits = checkHandle.good();
  checkHandle.close();
  std::ofstream fileHandle(fileName, std::ios::app);
  if (fileHandle.good()) {
    if (!fileExits) {
      fileHandle << "Symbol,Company,Loss,Epochs,Duration\n";
    }
    fileHandle << modelName << "," << companyName << "," << lossVal << ","
               << epoch << "," << time << '\n';
  }
  fileHandle.close();
}

std::vector<float> extractData(torch::Tensor &tensorData) {
  std::vector<float> result;
  for (int64_t idx = 0; idx < tensorData.size(0); ++idx) {
    result.push_back(tensorData[idx].item<float>());
  }
  return result;
}
} // namespace

NetworkTrainer::NetworkTrainer(int64_t input, int64_t hidden, int64_t output,
                               int64_t numLayers, int64_t prevSamples,
                               double learningRate, int64_t maxEpochs,
                               const std::string &modelName,
                               const std::string &companyName)
    :

      gpuAvailable(torch::cuda::is_available()), maxEpochs(maxEpochs),
      prevSamples(prevSamples), modelName(modelName), companyName(companyName),
      lstmNetwork(nullptr), optimizer(nullptr)

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
  linearOpts.with_bias(true);
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

std::vector<float> NetworkTrainer::fit(const std::vector<float> &x_train,
                                       const std::vector<float> &y_train,
    const std::vector<float> &x_test,
    const std::vector<float> &y_test) {

  torch::Tensor y_pred, input, target, input_test, target_test;
  torch::Tensor loss;

  bool noValidateDataSet = x_test.empty() && y_test.empty();
  bool saveFlag = false;
  input = torch::tensor(x_train).view({prevSamples, -1, 1});
  target = torch::tensor(y_train);

  if (!noValidateDataSet) {
      input_test = torch::tensor(x_test).view({ prevSamples, -1, 1 });
      target_test = torch::tensor(y_test);
  }
  if (gpuAvailable) {
    input = input.to(torch::kCUDA);
    target = target.to(torch::kCUDA);
    if (!noValidateDataSet) {
        input_test = input_test.to(torch::kCUDA);
        target_test = target_test.to(torch::kCUDA);
    }
  }

  float running_loss = 1, minimumLoss = 1, training_loss = 1;
  int64_t epoch = 0;
  input.set_requires_grad(true);

  Timer t1("Total Elapsed Time: %2.f\n");
  Timer t2("Epoch Time: %.2f ");

  const std::string neuralNetLogFile =
      NetworkConstants::kRootFolder + modelName + ".pt";
  const std::string predictLogFile =
      NetworkConstants::kRootFolder + modelName + "_pred.csv";

  if (!modelName.empty()) {
    try {
      loadModel(neuralNetLogFile);
      std::cout << "Loaded a existing trained model " << neuralNetLogFile
                << "\n";
    } catch (...) {
      std::cout << "Starting a fresh training...\n";
    }
  }

 torch::optim::LBFGSOptions lbfgs(NetworkConstants::kLearningRate);
 auto lbfgsOptimizer = std::make_shared<torch::optim::LBFGS>(torch::optim::LBFGS(lstmNetwork->parameters(), lbfgs));

 auto closure = [this, &lbfgsOptimizer, &y_pred, &loss, &input, &target]() {
      lbfgsOptimizer->zero_grad();
      y_pred = lstmNetwork->forward(input);
      loss = torch::mse_loss(y_pred, target);
      loss.backward();
      return loss;
  };

  while (running_loss > kRunningLoss) {
#ifndef TEST
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
#else
      lbfgsOptimizer->step(closure);
#endif
      training_loss = loss.item<float>();
      if (noValidateDataSet) {
          running_loss = loss.item<float>();
          // Save model and write the predicted tensor
          if (minimumLoss > running_loss) {
              saveModel(neuralNetLogFile);
              dataWriter(predictLogFile, extractData(y_pred));
              minimumLoss = running_loss;
          }
      }
      else 
     {
       // Validate and save model
        torch::NoGradGuard nograd;
        auto validateOut = lstmNetwork->forward(input_test);
        auto validateLoss = torch::mse_loss(validateOut, target_test);
        running_loss = validateLoss.item<float>();
        saveFlag = false;
        if (minimumLoss > running_loss) {
            saveModel(neuralNetLogFile);
            dataWriter(predictLogFile, extractData(y_pred));
            minimumLoss = running_loss;
            saveFlag = true;
        }
    }
    if (epoch >= this->maxEpochs || t2 > NetworkConstants::kMaxTrainTime) {
      std::cout << "Cannot converge after epoch " << epoch << ": "
                << minimumLoss << std::endl;
      return extractData(y_pred.detach());
    }

    else if (running_loss < NetworkConstants::kMinimumLoss) {
      std::cout << "Network fully trained!\n";
      logFullyTrainedModel(modelName, companyName, running_loss, epoch, t2);
      return extractData(y_pred.detach());
    }

    else if (epoch % 10 == 0 || saveFlag) {
        t2.show(false);
        std::cout << " epoch " << epoch << " [Traing Loss = " << training_loss
            << " Validation Loss = " << running_loss
            << " ( "
            << companyName << " ) ]" << ((saveFlag) ? "(**best**)\n" : "\n");
    }
    epoch++;
  }

  return extractData(y_pred.detach());
}

void NetworkTrainer::saveModel(const std::string &fileName) const {
  torch::save(lstmNetwork, fileName);
}

void NetworkTrainer::loadModel(const std::string &fileName) {
  torch::load(lstmNetwork, fileName);
}
