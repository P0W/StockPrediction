/*
 * StockPrices.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef STOCKPRICES_HPP_
#define STOCKPRICES_HPP_

#include "ITimeSeries.hpp"
#include <memory>
#include <string>
#include <tuple>
#include <vector>

template <typename T> class MinMaxScaler;

class StockData {
public:
  explicit StockData(const std::string &date, float val);
  ~StockData() = default;
  StockData(const StockData &) = default;
  StockData &operator=(const StockData &) = default;

  bool operator<(const StockData &other);

  float getClosePrice() const;
  std::string getDate() const;
  bool wasBadEntry() const;

private:
  std::string m_date;
  float m_closePrice;
  uint64_t m_dateNum;
  bool m_hasError;
};

class StockPrices : public ITimeSeries {
public:
  explicit StockPrices(MinMaxScaler<float> &scaler);
  virtual ~StockPrices();
  StockPrices(const StockPrices &) = delete;
  StockPrices &operator=(const StockPrices &) = delete;

  virtual bool loadTimeSeries(const std::string &stockSymbol) override;
  virtual void reshapeSeries(float testSplitRatio,
                             int64_t num_prev_samples) override;
  virtual void normalizeData() override;

  std::tuple<std::vector<float>, std::vector<float>, std::vector<std::string>>
  getTrainData() const;

private:
  MinMaxScaler<float> &scaler;
  std::vector<float> stockClosePrices;
  std::vector<float> normalizedStockClosePrices;
  std::vector<std::string> dates;

  std::vector<float> x_train;
  std::vector<float> y_train;
};

#endif /* STOCKPRICES_HPP_ */
