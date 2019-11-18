/*
 * StockPrices.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef STOCKPRICES_HPP_
#define STOCKPRICES_HPP_

#include "ITimeSeries.hpp"
#include "NetworkConstants.hpp"
#include <memory>
#include <string>
#include <tuple>
#include <vector>

/**
 * Forward declarations for MinMaxScaler class
 */
template <typename T> class MinMaxScaler;

/**
 * @class StockData
 * @brief A class responsible for processing a single stock price and dates for
 * those stock prices. This class will <ul> <li> Validate the stock date. <li>
 * Validate the stock price. <li> Indicate whether a stock price set has bad
 * entry. <li> Provide a means to compare two stock price with respect to date.
 * </ul>
 */
class StockData {
public:
  /**
   * @brief Constructor
   * @param [in] date The date for a given stock price.
   * @param [in] val The stock price.
   */
  explicit StockData(const std::string &date, float val);
  /**
   * @brief Default Destructor
   */
  ~StockData() = default;
  /**
   * @brief Default Copy Constructor
   */
  StockData(const StockData &) = default;
  /**
   * @brief Default Copy Assignment Operator
   */
  StockData &operator=(const StockData &) = default;
  /**
   * @brief Overloaded Comparison operator to compare two dates
   * @param [in] other The other StockData object to compare with this current
   * StockData object.
   */
  bool operator<(const StockData &other);
  /**
   * @brief Method to get the stock close price
   */
  float getClosePrice() const;
  /**
   * @brief Method to get the stock close price date
   */
  std::string getDate() const;
  /**
   * @brief Method to indicate whether the stock price or date is invalid
   */
  bool wasBadEntry() const;

private:
  std::string m_date; ///< The date for the stock price
  float
      m_closePrice; ///< The closing price of the stock price at the given date
  uint64_t m_dateNum; ///< The interger format of date, suitable for handy
                      ///< comparisons
  bool m_hasError;    ///< A flag to indicate is the give stock data is invalid
};

/**
 * @class StockPrices
 * @brief A class responsible for managing stock prices and dates for those
 * stock prices. This class will <ul> <li> Download the historical stock prices
 * from csv file. <li> Normalize the stock prices in [0,1] range. <li> Re-shape
 * and split the stock dataset into test and validation set.
 * </ul>
 */
class StockPrices : public ITimeSeries {
public:
  /**
   * @brief Constructor
   * @param [in] scaler The reference to MinMaxScaler object.
   */
  explicit StockPrices(MinMaxScaler<float> &scaler);
  /**
   * @brief Default Destructor
   */
  virtual ~StockPrices();
  /**
   * @brief Deleted Copy Constructor
   */
  StockPrices(const StockPrices &) = delete;
  /**
   * @brief Deleted Copy Assignment Operator
   */
  StockPrices &operator=(const StockPrices &) = delete;
  /**
   * @brief Method to load raw stock data from quandl APIs, through csv files
   * @param [in] stockSymbol The BSE stock symbol for which historical stock
   * prices needs to be loaded.
   * @param [in] maxDataSize The maximum number of stock prices to take into
   * account.
   */
  virtual bool loadTimeSeries(
      const std::string &stockSymbol,
      const size_t &maxDataSize = NetworkConstants::kMaxStockPrices) override;
  /**
   * @brief Method to split and reshape stock data
   * @param [in] testSplitRatio The ratio used for training and test dataset.
   * @param [in] num_prev_samples The number of past sample used for creation of
   * test and training dataset.
   */
  virtual void reshapeSeries(float testSplitRatio,
                             int64_t num_prev_samples) override;
  /**
   * @brief Method to normalize the dataset
   */
  virtual void normalizeData() override;

  /**
   * @brief Method to return the training dataset
   */
  std::tuple<std::vector<float>, std::vector<float>, std::vector<std::string>>
  getTrainData() const;

  /**
   * @brief Method to return the validation dataset
   */
  std::tuple<std::vector<float>, std::vector<float>, std::vector<std::string>>
  getTestData() const;

private:
  MinMaxScaler<float> &scaler; ///< The reference to MinMaxScaler float object
  std::vector<float> stockClosePrices; ///< The vector of stock close price
  std::vector<float> normalizedStockClosePrices; ///< The vector of normalized
                                                 ///< stock close price
  std::vector<std::string> dates; ///< The vector of stock close price dates

  std::vector<std::string> dates_train; ///< The vector of train dataset dates
  std::vector<std::string> dates_test;  ///< The vector of test dataset dates

  std::vector<float>
      x_train; ///< The vector of reshaped normalized train dataset input
  std::vector<float>
      y_train; ///< The vector of reshaped normalized train dataset ouput
  std::vector<float>
      x_test; ///< The vector of reshaped normalized test dataset ouput
  std::vector<float>
      y_test; ///< The vector of reshaped normalized test dataset ouput
};

#endif /* STOCKPRICES_HPP_ */
