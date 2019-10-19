#ifndef STOCKPREDICTOR_HPP_
#define STOCKPREDICTOR_HPP_

#include "StockLSTM.hpp"
#include <string>

class StockPredictor {
public:
  StockPredictor();

  void loadModel(const std::string &stockSymbol);
  void predict(const int64_t N);
  void testModel();
  virtual ~StockPredictor();

private:
  void loadTimeSeries();
  std::shared_ptr<StockLSTM> m_lstmNetwork;
  std::string m_stockSymbol;
};

#endif //! STOCKPREDICTOR_HPP_