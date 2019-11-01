/*
 * NetworkConstants.cpp
 *
 *  Created on: 20-Oct-2019
 *      Author: Prashant Srivastava
 */

#include "StockPredictor.hpp"
#include "NetworkConstants.hpp"
#include "StockLSTM.hpp"
#include "StockPrices.hpp"
#include <cassert>
#include <iostream>
#include <torch/torch.h>

StockPredictor::StockPredictor()
    : m_minmaxScaler{}, m_lstmNetwork(nullptr),
      m_stockPrices(nullptr), m_stockSymbol{} {
  torch::nn::LSTMOptions lstmOpts1(NetworkConstants::input_size,
                                   NetworkConstants::hidden_size);
  torch::nn::LSTMOptions lstmOpts2(NetworkConstants::hidden_size,
                                   NetworkConstants::hidden_size);
  lstmOpts1.layers(NetworkConstants::num_of_layers)
      .dropout(NetworkConstants::klsmt1DropOut)
      .with_bias(NetworkConstants::kLstmIncludeBias);
  lstmOpts2.layers(NetworkConstants::num_of_layers)
      .dropout(NetworkConstants::klsmt2DropOut)
      .with_bias(NetworkConstants::kLstmIncludeBias);

  torch::nn::LinearOptions linearOpts(NetworkConstants::hidden_size,
                                      NetworkConstants::output_size);
  linearOpts.with_bias(false);
  torch::nn::DropoutOptions dropOutOpts(NetworkConstants::kdropOutDropOut);
  m_lstmNetwork = std::make_shared<StockLSTM>(lstmOpts1, lstmOpts2, dropOutOpts, linearOpts);
}

void StockPredictor::loadModel(const std::string &stockSymbol) {
  if (!stockSymbol.empty()) {
    m_stockSymbol = stockSymbol;
    const std::string trainedModel = m_stockSymbol + ".pt";
    try {
      m_stockPrices.reset(new StockPrices(m_minmaxScaler));

      if (m_stockPrices->loadTimeSeries(m_stockSymbol)) {
        std::cout << m_stockSymbol << " has one or more bad entries\n";
      } else {

        m_stockPrices->normalizeData();
        m_stockPrices->reshapeSeries(NetworkConstants::kSplitRatio,
                                     NetworkConstants::kPrevSamples);
        torch::load(m_lstmNetwork, trainedModel);
      }
      std::cout << "Loaded..." << trainedModel << '\n';
    } catch (...) {
      std::cout << "Failed to load" << trainedModel << '\n';
    }
  }
}

void StockPredictor::predict(const int64_t N) {
  // N is the next future N days predictions
  // For predicting 1 sample we would need last N but
  // NetworkConstants::kPrevSamples samples
  auto testData = m_stockPrices->getTrainData();
  auto testSamples = std::get<0>(testData);

  // Erase all but  NetworkConstants::kPrevSamples samples
  testSamples.erase(std::begin(testSamples),
                    std::end(testSamples) - NetworkConstants::kPrevSamples);
  assert(testSamples.size() ==
         static_cast<size_t>(NetworkConstants::kPrevSamples));

  std::vector<float> predictedNormalizedPrices;

  for (int64_t t = 0; t < N; ++t) {
    const auto &nextClosingNormalizedPrices = predict(testSamples);

    predictedNormalizedPrices.push_back(nextClosingNormalizedPrices[0]);

    // Prepare for next training set
    testSamples.erase(testSamples.begin());
    testSamples.push_back(nextClosingNormalizedPrices[0]);

    assert(testSamples.size() ==
           static_cast<size_t>(NetworkConstants::kPrevSamples));
  }

  assert(predictedNormalizedPrices.size() == static_cast<size_t>(N));

  fileLogger(m_stockSymbol + "_future.csv", predictedNormalizedPrices);
}

void StockPredictor::testModel(const std::string &args) {

  const std::string testPreditorLogFile = m_stockSymbol + "_test_pred.csv";
  const std::string testLogFile = m_stockSymbol + "_test.csv";
  int64_t days = -1;
  bool grabTrainData = false;
  try {
    days = std::stoi(args);
  } catch (...) {
    days = -1;
  }
  if (days != -1) {
    predict(days);
    return;
  }
  if (args.compare("trainData") == 0) {
    grabTrainData = true;
  }
  const auto &dataSet = (grabTrainData) ? m_stockPrices->getTrainData()
                                        : m_stockPrices->getTestData();

  const auto &x_test = std::get<0>(dataSet);
  const auto &y_test = std::get<1>(dataSet);
  const auto &allDates = std::get<2>(dataSet);

  std::ofstream fileHandle(testLogFile);
  fileHandle << "Close, Price\n";
  std::transform(std::cbegin(y_test), std::cend(y_test), std::cbegin(allDates),
                 std::ostream_iterator<std::string>(fileHandle, "\n"),
                 [](const auto &price, const auto &date) {
                   return std::string(date) + std::string(",") +
                          std::to_string(price);
                 });
  fileHandle.close();

  // Predict the output using the neural network from test dataSet
  if (m_lstmNetwork) {
    const auto &y_test_pred = predict(x_test, y_test);

    std::cout << "WEBREQUEST Writing test dataset to " << testPreditorLogFile
              << '\n';
    fileLogger(testPreditorLogFile, y_test_pred, y_test, allDates);
    // fileLogger(testLogFile, y_test, allDates);
  } else {
    std::cout << "WEBREQUEST Cannnot predict data. \n";
  }
}

StockPredictor::~StockPredictor() {}

void StockPredictor::loadTimeSeries() {}

std::vector<float>
StockPredictor::predict(const std::vector<float> &input,
                        const std::vector<float> &expectedOuput) {

  std::vector<float> result;
  torch::Tensor x_test = torch::tensor(input), target;
  float accumulated_loss = 0.0f;

  bool gpuAvailable = torch::cuda::is_available();
  if (gpuAvailable) {
    m_lstmNetwork->to(torch::kCUDA);
  } else {
    m_lstmNetwork->to(torch::kCPU);
  }
  if (!expectedOuput.empty()) {
      target = torch::tensor(expectedOuput)
          .to(gpuAvailable ? torch::kCUDA : torch::kCPU);
  }
  x_test = x_test.view({NetworkConstants::kPrevSamples, -1, 1});
  x_test = x_test.to(gpuAvailable ? torch::kCUDA : torch::kCPU);

  m_lstmNetwork->eval();
  {
    torch::NoGradGuard no_grad;

    if (input.size() < 100 * NetworkConstants::kPrevSamples) {
        for (int64_t idx = 0; idx < x_test.size(1); ++idx) {
            const auto &slicedTensor = x_test.slice(1, idx, idx + 1);
            auto validateOut = m_lstmNetwork->forward(slicedTensor);
            result.push_back(validateOut.item<float>());
            if (!expectedOuput.empty()) {
                const auto &slicedTargetTensor = target.slice(0, idx, idx + 1);
                auto validateLoss = torch::mse_loss(validateOut, slicedTargetTensor);
                accumulated_loss += std::pow(validateLoss.item<float>(), 2.0f);
            }
        }
        if (!expectedOuput.empty()) {
            accumulated_loss = std::sqrt(accumulated_loss / x_test.size(1));
            std::cout << "WEBREQUEST Prediction Loss: " << accumulated_loss << '\n';
        }
    }
    else {
        auto validateOut = m_lstmNetwork->forward(x_test);
        for (size_t ele = 0; ele < x_test.size(1); ++ele) {
            result.push_back(validateOut[ele].item<float>());
        }
        if (!expectedOuput.empty()) {
            auto validateLoss = torch::mse_loss(validateOut, target);
            std::cout << "WEBREQUEST Prediction Loss: " << validateLoss.item<float>() << '\n';
        }
    }
  }

  return result;
}

void StockPredictor::fileLogger(
    const std::string &logFileName, const std::vector<float> &y_test,
    const std::vector<float> &input,
    const std::vector<std::string> &allDates) const {
  std::ofstream fileHandle(logFileName, std::ios::trunc);
  if (!allDates.empty()) {
    fileHandle << "date,price,actual_price\n";
  } else {
    fileHandle << "price\n";
  }
  if (fileHandle.good()) {
    for (size_t idx = 0; idx < y_test.size(); ++idx) {
      if (y_test[idx] >= 0.0 && y_test[idx] <= 1.0) {

      } else {
        std::cout << "Wrong entry at : " << idx << " :" << y_test[idx] << '\n';
      }
      if (!allDates.empty()) {
        fileHandle << allDates[idx] << "," << m_minmaxScaler(y_test[idx]) << ","
                   << m_minmaxScaler(input[idx]) << '\n';
      } else {
        fileHandle << m_minmaxScaler(y_test[idx]) << '\n';
      }
    }
  }
  fileHandle.close();
}
