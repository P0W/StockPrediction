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
      .with_bias(NetworkConstants::kIncludeBias);
  lstmOpts2.layers(NetworkConstants::num_of_layers)
      .dropout(NetworkConstants::klsmt2DropOut)
      .with_bias(NetworkConstants::kIncludeBias);

  torch::nn::LinearOptions linearOpts(NetworkConstants::hidden_size,
                                      NetworkConstants::output_size);
  linearOpts.with_bias(false);
  m_lstmNetwork = std::make_shared<StockLSTM>(lstmOpts1, lstmOpts2, linearOpts);
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

void StockPredictor::testModel() {

  const std::string testPreditorLogFile = m_stockSymbol + "_test_pred.csv";
  const std::string testLogFile = m_stockSymbol + "_test.csv";

  const auto &testData = m_stockPrices->getTestData();

  const auto &x_test = std::get<0>(testData);
  const auto &y_test = std::get<1>(testData);
  const auto &allDates = std::get<2>(testData);

  // Predict the output using the neural network from test dataSet
  if (m_lstmNetwork) {
    const auto &y_test_pred = predict(x_test);

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

std::vector<float> StockPredictor::predict(const std::vector<float> &input) {

  std::vector<float> result;
  auto x_test = torch::tensor(input);
  x_test = x_test.view({NetworkConstants::kPrevSamples, -1, 1});
  x_test = x_test.to(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  auto pred = m_lstmNetwork->forward(x_test);
  for (int64_t idx = 0; idx < pred.size(0); ++idx) {
    result.push_back(pred[idx].item<float>());
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
