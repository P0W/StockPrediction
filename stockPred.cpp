/*
 * stockPred.cpp
 *
 *  Created on: 23-Sep-2019
 *      Author: Prashant Srivastava
 */

#include <memory>
#include <fstream>

#include "MinMaxScaler.hpp"
#include "NetworkTrainer.hpp"
#include "StockPrices.hpp"

#include "NetworkConstants.hpp"
#include "csv.h"

namespace {
class StockNetworkTrainer : public NetworkTrainer {
 public:
  StockNetworkTrainer(const std::string& fileName,
                      const MinMaxScaler<float>& minmaxScaler,
      const std::vector<std::string> allDates)
      : NetworkTrainer(
            NetworkConstants::input_size, NetworkConstants::hidden_size,
            NetworkConstants::output_size, NetworkConstants::num_of_layers,
            NetworkConstants::kPrevSamples, NetworkConstants::kLearningRate,
            NetworkConstants::kMaxEpochs,
            NetworkConstants::kRootFolder + fileName),
      minMaxScaler{ minmaxScaler }, allDates{ allDates } {}

  virtual void dataWriter(const std::string& logFile, const torch::Tensor& tensorData) override {
      std::ofstream fileHandle(logFile, std::ios::trunc);
      fileHandle << "date,price\n";
      if (fileHandle.good()) {
          for (int64_t idx = 0; idx < tensorData.size(0); ++idx) {
              fileHandle << allDates.at(idx)
                  << ","
                  << minMaxScaler(tensorData[idx].item<float>()) 
                  << '\n';
          }
      }

      fileHandle.close();
  }

 private:
  std::string fileName;
  const MinMaxScaler<float>& minMaxScaler;
  std::vector<std::string> allDates;
};

void updateConfig(const std::string& configFileName, 
    const std::string& stockSymbol, 
    const std::string& stockName = "") {
    std::ifstream testFileHandle(configFileName);
    bool isPresent = testFileHandle.good();
    testFileHandle.close();

    std::ofstream fileHandle(configFileName, std::ios::app);

    if (!isPresent) {
        fileHandle << "Symbol,Company\n";
    }
    fileHandle << stockSymbol << ',' << stockName << '\n';
    fileHandle.close();
}

bool NetworkTrainerFacade(
    const std::string& stockSymbol, 
    const std::string& companyName = "") {
    
  MinMaxScaler<float> minmaxScaler;
  StockPrices stockData(minmaxScaler);
  if (stockData.loadTimeSeries(stockSymbol)) {
      std::cout << stockSymbol << ":" << companyName << " has one or more bad entries\n";
      return false;
  }

  stockData.normalizeData();
  stockData.reshapeSeries(NetworkConstants::kSplitRatio, NetworkConstants::kPrevSamples);

  auto trainData = stockData.getTrainData();

  // Convert values to Pytorch Tensors
  torch::Tensor x_train = torch::tensor(std::get<0>(trainData));
  torch::Tensor y_train = torch::tensor(std::get<1>(trainData));

  // Record this stock for front end to update
  updateConfig(NetworkConstants::kRootFolder + "stock_train.csv", stockSymbol, companyName);

  std::shared_ptr<NetworkTrainer> model =
      std::make_shared<StockNetworkTrainer>(stockSymbol, minmaxScaler, std::get<2>(trainData));

  model->dataWriter(NetworkConstants::kRootFolder + stockSymbol + "_train.csv", y_train);

  torch::Tensor y_pred = model->fit(x_train, y_train);

  return true;
}
}  // namespace

int main(int argc, char** argv) {
  std::string stockSymbol;

  if (argc != 2) {
    std::cout << "Missing Stock Symbol...reading top 100 BSE stocks\n";
    const std::string bse100File = NetworkConstants::kRootFolder + "BSE100.csv";
    std::string companyName;
    io::CSVReader<2> in(bse100File);
    in.read_header(io::ignore_extra_column, "Symbol", "Name");

    while (argc >= 1 && in.read_row(stockSymbol, companyName)) {
        if (!NetworkTrainerFacade(stockSymbol, companyName)) {
            continue;
        }
    }
  } else {
    (void)NetworkTrainerFacade(argv[1]);
  }

  return 0;
}
