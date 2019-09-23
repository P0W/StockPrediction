/*
 * StockPrices.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef STOCKPRICES_HPP_
#define STOCKPRICES_HPP_

#include <memory>
#include <string>
#include <vector>
#include "ITimeSeries.hpp"

template <typename T>
class MinMaxScaler;

class StockData {
 public:
  explicit StockData(std::string date, float val);
  ~StockData() = default;
  StockData(const StockData&) = default;
  StockData& operator=(const StockData&) = default;

  bool operator<(const StockData& other);

  float getClosePrice() const;

 private:
  std::string date;
  float closePrice;
};

class StockPrices : public ITimeSeries {
 public:
  explicit StockPrices(const std::string& stockSymbol, int64_t prevSamples,
                       MinMaxScaler<float>& scaler);
  virtual ~StockPrices();
  StockPrices(const StockPrices&) = delete;
  StockPrices& operator=(const StockPrices&) = delete;

  virtual void loadTimeSeries() override;
  virtual void reshapeSeries(float testSplitRatio) override;
  virtual void normalizeData() override;

  std::pair<std::vector<float>, std::vector<float>> getTrainData() const;

 private:
  std::string stockSymbol;
  int64_t num_prev_samples;
  MinMaxScaler<float>& scaler;
  std::vector<float> stockClosePrices;
  std::vector<float> normalizedStockClosePrices;

  std::vector<float> x_train;
  std::vector<float> y_train;
};

#endif /* STOCKPRICES_HPP_ */
