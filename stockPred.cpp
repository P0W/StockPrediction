

#include <algorithm>
#include <memory>
#include "MinMaxScaler.hpp"
#include "NetworkTrainer.hpp"
#include "StockPrices.hpp"

#include "NetworkConstants.hpp"
#include "csv.h"

namespace {
class StockNetworkTrainer : public NetworkTrainer {
 public:
  StockNetworkTrainer(const std::string& fileName,
                      const MinMaxScaler<float>& minmaxScaler)
      : NetworkTrainer(
            NetworkConstants::input_size, NetworkConstants::hidden_size,
            NetworkConstants::output_size, NetworkConstants::num_of_layers,
            NetworkConstants::prevSamples, NetworkConstants::learningRate,
            NetworkConstants::maxEpochs,
            NetworkConstants::rootFolder + fileName + ".pt"),
        fileName(NetworkConstants::rootFolder + fileName + "_pred.txt"),
        minMaxScaler{minmaxScaler} {}

  virtual void dataWriter(const torch::Tensor& tensorData) override {
    NetworkTrainer::saveTensor(tensorData, fileName, minMaxScaler);
  }

 private:
  std::string fileName;
  const MinMaxScaler<float>& minMaxScaler;
};
}  // namespace

int main(int argc, char** argv) {
  std::string stockSymbol;

  if (argc != 2) {
    std::cout << "Missing Stock Symbol...reading top 100 BSE stocks\n";

  } else {
    stockSymbol = argv[1];
  }

  const std::string bse100File =
      "/home/powprashant/Downloads/example-app/BSE100.csv";

  MinMaxScaler<float> minmaxScaler;

  io::CSVReader<1> in(bse100File);
  in.read_header(io::ignore_extra_column, "Symbol");

  while (argc >= 1 && in.read_row(stockSymbol)) {
    StockPrices stockData(stockSymbol, NetworkConstants::prevSamples,
                          minmaxScaler);
    stockData.loadTimeSeries();
    stockData.normalizeData();
    stockData.reshapeSeries(NetworkConstants::splitRatio);

    auto trainData = stockData.getTrainData();

    // Convert values to Pytorch Tensors
    torch::Tensor x_train = torch::tensor(trainData.first);
    torch::Tensor y_train = torch::tensor(trainData.second);
    torch::Tensor y_pred;

    NetworkTrainer::saveTensor(
        y_train, NetworkConstants::rootFolder + stockSymbol + "_train.txt",
        minmaxScaler);

    std::shared_ptr<NetworkTrainer> model =
        std::make_shared<StockNetworkTrainer>(stockSymbol, minmaxScaler);

    y_pred = model->fit(x_train, y_train);
  }
}
