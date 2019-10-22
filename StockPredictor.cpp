#include "StockPredictor.hpp"
#include "NetworkConstants.hpp"
#include <cassert>
#include <iostream>

StockPredictor::StockPredictor() : m_minmaxScaler{}, m_lstmNetwork(nullptr), m_stockPrices(nullptr), m_stockSymbol{} {
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
      }
      else {
          
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
    // For predicting 1 sample we would need last N but NetworkConstants::kPrevSamples samples
    auto testData = m_stockPrices->getTestData();
    auto testSamples = std::get<0>(testData);

    // Erase all but  NetworkConstants::kPrevSamples samples
    testSamples.erase(std::begin(testSamples), std::end(testSamples) - NetworkConstants::kPrevSamples);
    assert(testSamples.size() == NetworkConstants::kPrevSamples);

    std::vector<float> predictedPrices;

    for (int64_t t = 0; t < N; ++t) {
        auto nextClosingPriceTensor = predict(testSamples);
        float nextClosingPrice = m_minmaxScaler(nextClosingPriceTensor[0].item<float>());
        predictedPrices.push_back(nextClosingPrice);

        // Prepare for next training set
        testSamples.erase(testSamples.begin());
        testSamples.push_back(nextClosingPrice);

        assert(testSamples.size() == NetworkConstants::kPrevSamples);
    }

    assert(predictedPrices.size() == N);

    // Fix Me !
    fileLogger(m_stockSymbol + "_future.csv", std::get<2>(testData), predictedPrices);
}

void StockPredictor::testModel() {
  
  const std::string testPreditorLogFile = m_stockSymbol + "_test.csv";


  auto testData = m_stockPrices->getTestData();

  auto allDates = std::get<2>(testData);

  // Predict the output using the neural network from test dataSet
  if (m_lstmNetwork) {
    std::cout << "WEBREQUEST Calling forward on trained model for testset \n";

    torch::Tensor y_test_pred = predict(std::get<0>(testData));

    std::cout << "WEBREQUEST Writing test dataset to " << testPreditorLogFile
              << '\n';
    fileLogger(testPreditorLogFile, allDates, y_test_pred);
  } else {
    std::cout << "WEBREQUEST Cannnot predict data. \n";
  }
}

StockPredictor::~StockPredictor() {}

void StockPredictor::loadTimeSeries() {}

torch::Tensor StockPredictor::predict(const std::vector<float>& input)
{
    
    auto x_test = torch::tensor(input);
    x_test = x_test.view({ NetworkConstants::kPrevSamples, -1, 1 });
    x_test =
        x_test.to(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    auto pred = m_lstmNetwork->forward(x_test);

    return pred;
}

void StockPredictor::fileLogger(const std::string & logFileName, const std::vector<std::string>& allDates, const torch::Tensor& y_test) const
{
    std::ofstream fileHandle(logFileName, std::ios::trunc);
    fileHandle << "date,price\n";
    if (fileHandle.good()) {
        for (int64_t idx = 0; idx < y_test.size(0); ++idx) {
            fileHandle << allDates.at(idx) << ","
                << m_minmaxScaler(y_test[idx].item<float>()) << '\n';
        }
    }

    fileHandle.close();
}

void StockPredictor::fileLogger(const std::string & logFileName, const std::vector<std::string>& allDates, const std::vector<float>& y_test) const
{
    std::ofstream fileHandle(logFileName, std::ios::trunc);
    fileHandle << "date,price\n";
    if (fileHandle.good()) {
        for (int64_t idx = 0; idx < y_test.size(); ++idx) {
            fileHandle << allDates.at(idx) << ","
                << m_minmaxScaler(y_test[idx]) << '\n';
        }
    }
    fileHandle.close();
}
