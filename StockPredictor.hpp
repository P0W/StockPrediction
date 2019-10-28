/*
 * NetworkConstants.cpp
 *
 *  Created on: 20-Oct-2019
 *      Author: Prashant Srivastava
 */

#ifndef STOCKPREDICTOR_HPP_
#define STOCKPREDICTOR_HPP_

#include "MinMaxScaler.hpp"
#include <memory>
#include <string>
#include <vector>

class StockLSTM;
class StockPrices;

class StockPredictor {
public:
  StockPredictor();

  void loadModel(const std::string &stockSymbol);
  void predict(const int64_t N);
  void testModel();
  virtual ~StockPredictor();

private:
  void loadTimeSeries();
  std::vector<float> predict(const std::vector<float> &input);
  void fileLogger(const std::string &logFileName,
                  const std::vector<float> &y_test, 
                  const std::vector<float> &input = {}, 
      const std::vector<std::string> &allDates = {}) const;
  MinMaxScaler<float> m_minmaxScaler;
  std::shared_ptr<StockLSTM> m_lstmNetwork;
  std::shared_ptr<StockPrices> m_stockPrices;
  std::string m_stockSymbol;
};

#endif //! STOCKPREDICTOR_HPP_