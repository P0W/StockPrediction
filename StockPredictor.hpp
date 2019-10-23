#ifndef STOCKPREDICTOR_HPP_
#define STOCKPREDICTOR_HPP_


#include "MinMaxScaler.hpp"
#include <string>

class StockLSTM;
class StockPrices;

namespace torch {
    class Tensor;
}

class StockPredictor {
public:
  StockPredictor();

  void loadModel(const std::string &stockSymbol);
  void predict(const int64_t N);
  void testModel();
  virtual ~StockPredictor();

private:
  void loadTimeSeries();
  std::shared_ptr<torch::Tensor> predict(const std::vector<float>& input);
  void fileLogger(const std::string & logFileName, const std::vector<std::string>& allDates, const torch::Tensor& y_test) const;
  void fileLogger(const std::string & logFileName, const std::vector<std::string>& allDates, const std::vector<float>& y_test) const;
  MinMaxScaler<float> m_minmaxScaler;
  std::shared_ptr<StockLSTM> m_lstmNetwork;
  std::shared_ptr<StockPrices> m_stockPrices;
  std::string m_stockSymbol;
};

#endif //! STOCKPREDICTOR_HPP_