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

NetworkTrainer::NetworkTrainer(const std::string &modelName,
                               const std::string &companyName)
    :

      gpuAvailable(torch::cuda::is_available()), 
      modelName(modelName), companyName(companyName),
      lstmNetwork(nullptr), optimizer(nullptr)

{
  torch::nn::LSTMOptions lstmOpts1(NetworkConstants::input_size, NetworkConstants::hidden_size);
  torch::nn::LSTMOptions lstmOpts2(NetworkConstants::hidden_size, NetworkConstants::hidden_size);
  lstmOpts1.layers(NetworkConstants::num_of_layers)
      .dropout(NetworkConstants::klsmt1DropOut)
      .with_bias(NetworkConstants::kLstmIncludeBias);
  lstmOpts2.layers(NetworkConstants::num_of_layers)
      .dropout(NetworkConstants::klsmt2DropOut)
      .with_bias(NetworkConstants::kLstmIncludeBias);

  torch::nn::LinearOptions linearOpts(NetworkConstants::hidden_size, NetworkConstants::output_size);
  linearOpts.with_bias(NetworkConstants::kIncludeLinearBias);
  torch::nn::DropoutOptions dropOutOpts(NetworkConstants::kdropOutDropOut);
  lstmNetwork = std::make_shared<StockLSTM>(lstmOpts1, lstmOpts2, dropOutOpts, linearOpts);

  // Create the optimizer in pytorch
  torch::optim::AdamOptions opts(NetworkConstants::kLearningRate);
  optimizer = std::make_shared<torch::optim::Adam>(
      torch::optim::Adam(lstmNetwork->parameters(), opts));

  // Check is GPU is Available, if it is move the network processing on GPU
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
  bool iValidatedGood = false;
  
  float running_loss = std::numeric_limits<float>::infinity(),
      minimumLoss = std::numeric_limits<float>::infinity(),
      training_loss = std::numeric_limits<float>::infinity(),
      accumulated_loss = 0.0;
  int64_t epoch = 0;
  
  const std::string neuralNetLogFile =
      NetworkConstants::kRootFolder + modelName + ".pt";
  const std::string predictLogFile =
      NetworkConstants::kRootFolder + modelName + "_pred.csv";

  input = torch::tensor(x_train).view({NetworkConstants::kPrevSamples, -1, 1});
  target = torch::tensor(y_train);

  if (!noValidateDataSet) {
    input_test = torch::tensor(x_test).view({ NetworkConstants::kPrevSamples, -1, 1});
    target_test = torch::tensor(y_test);
  }

  // Check if GPU is Available, if it is, move all input and target tensors on GPU
  if (gpuAvailable) {
    input = input.to(torch::kCUDA);
    target = target.to(torch::kCUDA);
    if (!noValidateDataSet) {
      input_test = input_test.to(torch::kCUDA);
      target_test = target_test.to(torch::kCUDA);
    }
  }

  input.set_requires_grad(true);

  Timer t1("Total Elapsed Time: %2.f\n");
  Timer t2("Epoch Time: %.2f ");

  if (!modelName.empty()) {
    try {
      loadModel(neuralNetLogFile);
      std::cout << "Loaded a existing trained model " << neuralNetLogFile
                << "\n";
      std::cout << *lstmNetwork << std::endl;
    } catch (...) {
      std::cout << "Starting a fresh training...\n";
    }
  }
#ifdef USELBFGS
  torch::optim::LBFGSOptions lbfgs(NetworkConstants::kLearningRate);
  auto lbfgsOptimizer = std::make_shared<torch::optim::LBFGS>(
      torch::optim::LBFGS(lstmNetwork->parameters(), lbfgs));
  auto closure = [this, &lbfgsOptimizer, &y_pred, &loss, &input, &target]() {
    lbfgsOptimizer->zero_grad();
    y_pred = lstmNetwork->forward(input);
    loss = torch::mse_loss(y_pred, target);
    loss.backward();
    return loss;
  };
#endif
  while (running_loss > kRunningLoss) {
    // Set network to training mode.
    lstmNetwork->train();
#ifndef USELBFGS
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
    // Calculate the training loss
    training_loss = loss.item<float>();
    if (noValidateDataSet) {
      running_loss = loss.item<float>();
      saveFlag = false;
      // Save model and write the predicted tensor
      if (minimumLoss > running_loss) {
        saveModel(neuralNetLogFile);
        dataWriter(predictLogFile, extractData(y_pred));
        minimumLoss = running_loss;
        saveFlag = true;
      }
    } else {
      // Validate and save model
      // Stop building gradients here
      torch::NoGradGuard nograd;
      // Set network to evaluation mode
      lstmNetwork->eval();
      accumulated_loss = 0.0;
      for (int64_t idx = 0; idx < input_test.size(1); ++idx) {
        const auto &slicedTensor = input_test.slice(1, idx, idx + 1);
        const auto &slicedTargetTensor = target_test.slice(0, idx, idx + 1);
        auto validateOut = lstmNetwork->forward(slicedTensor);
        auto validateLoss = torch::mse_loss(validateOut, slicedTargetTensor);
        accumulated_loss += std::pow(validateLoss.item<float>(), 2.0f);
      }
      // Calculate Mean Square Error
      running_loss = std::sqrt(accumulated_loss/ input_test.size(1));
      saveFlag = false;
      // If minimumLoss is greater than the running loss, save this trained model as a checkpoint
      if (minimumLoss > running_loss) {
        saveModel(neuralNetLogFile);
        dataWriter(predictLogFile, extractData(y_pred));
        minimumLoss = running_loss;
        saveFlag = true;
        iValidatedGood = true;
      }

      if (!saveFlag && (minimumLoss > training_loss)) {
        // Then also save model ... we don't want to discard good training
        saveModel(neuralNetLogFile);
        dataWriter(predictLogFile, extractData(y_pred));
        minimumLoss = training_loss;
        saveFlag = true;
        iValidatedGood = false;
      }

      // Set the network back to training mode
      lstmNetwork->train();
    }
    if (epoch >= NetworkConstants::kMaxEpochs || t2 > NetworkConstants::kMaxTrainTime) {
      std::cout << "Cannot converge after epoch " << epoch << ": "
                << minimumLoss << std::endl;
      return extractData(y_pred);
    }

    else if (running_loss < NetworkConstants::kMinimumLoss) {
      std::cout << "Network fully trained!\n";
      logFullyTrainedModel(modelName, companyName, running_loss, epoch, t2);
      return extractData(y_pred);
    }

    else if (epoch % 50 == 0 || saveFlag) {
      t2.show(false);
      std::cout << " epoch " << epoch << " [Traing Loss = " << training_loss
                << " Validation Loss = " << running_loss << " ( " << companyName
                << " ) ]" << ((saveFlag) ? "(**best " : "")
                << ((saveFlag && iValidatedGood)
                        ? " validation**)\n"
                        : (saveFlag && !iValidatedGood) ? " training**)\n"
                                                        : "\n");
    }
    epoch++;
  }

  return extractData(y_pred);
}

void NetworkTrainer::saveModel(const std::string &fileName) const {
  torch::save(lstmNetwork, fileName);
}

void NetworkTrainer::loadModel(const std::string &fileName) {
  torch::load(lstmNetwork, fileName);
}
