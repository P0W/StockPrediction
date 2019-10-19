#include "StockPredictor.hpp"
#include "MinMaxScaler.hpp"
#include "NetworkConstants.hpp"
#include "StockPrices.hpp"
#include <iostream>

StockPredictor::StockPredictor() : m_lstmNetwork(nullptr), m_stockSymbol{} {}

void StockPredictor::loadModel(const std::string &stockSymbol) {
  if (!stockSymbol.empty()) {
    m_stockSymbol = stockSymbol;
    const std::string trainedModel = m_stockSymbol + ".pt";
    try {
      torch::load(m_lstmNetwork, trainedModel);
      std::cout << "Loaded..." << trainedModel << '\n';
    } catch (...) {
      std::cout << "Failed to load" << trainedModel << '\n';
    }
  }
}

void StockPredictor::predict(const int64_t N) { (void)N; }

void StockPredictor::testModel() {
  MinMaxScaler<float> minmaxScaler;
  const std::string testPreditorLogFile = m_stockSymbol + "_test.csv";
  StockPrices stockData(minmaxScaler);
  if (stockData.loadTimeSeries(m_stockSymbol)) {
    std::cout << m_stockSymbol << " has one or more bad entries\n";
  }

  stockData.normalizeData();
  stockData.reshapeSeries(NetworkConstants::kSplitRatio,
                          NetworkConstants::kPrevSamples);

  auto testData = stockData.getTestData();

  auto allDates = std::get<2>(testData);

  // Convert values to Pytorch Tensors
  torch::Tensor x_test = torch::tensor(std::get<0>(testData));

  // Predict the output using the neural network from test dataSet
  torch::Tensor y_test_pred = m_lstmNetwork->forward(x_test);

  std::ofstream fileHandle(testPreditorLogFile, std::ios::trunc);
  fileHandle << "date,price\n";
  if (fileHandle.good()) {
    for (int64_t idx = 0; idx < y_test_pred.size(0); ++idx) {
      fileHandle << allDates.at(idx) << ","
                 << minmaxScaler(y_test_pred[idx].item<float>()) << '\n';
    }
  }

  fileHandle.close();
}

StockPredictor::~StockPredictor() {}

void StockPredictor::loadTimeSeries() {}
