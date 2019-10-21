#include "StockPredictor.hpp"
#include "MinMaxScaler.hpp"
#include "NetworkConstants.hpp"
#include "StockPrices.hpp"
#include <iostream>

StockPredictor::StockPredictor() : m_lstmNetwork(nullptr), m_stockSymbol{} {
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
  x_test = x_test.view({NetworkConstants::kPrevSamples, -1, 1});

  // Predict the output using the neural network from test dataSet
  if (m_lstmNetwork) {
    std::cout << "WEBREQUEST Calling forward on trained model for testset \n";

    x_test =
        x_test.to(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    torch::Tensor y_test_pred = m_lstmNetwork->forward(x_test);

    std::cout << "WEBREQUEST Writing test dataset to " << testPreditorLogFile
              << '\n';

    std::ofstream fileHandle(testPreditorLogFile, std::ios::trunc);
    fileHandle << "date,price\n";
    if (fileHandle.good()) {
      for (int64_t idx = 0; idx < y_test_pred.size(0); ++idx) {
        fileHandle << allDates.at(idx) << ","
                   << minmaxScaler(y_test_pred[idx].item<float>()) << '\n';
      }
    }

    fileHandle.close();
  } else {
    std::cout << "WEBREQUEST Cannnot predict data. \n";
  }
}

StockPredictor::~StockPredictor() {}

void StockPredictor::loadTimeSeries() {}
