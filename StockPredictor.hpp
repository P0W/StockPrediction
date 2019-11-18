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

/** 
* Forward Declaration for classes
*/
class StockLSTM;
class StockPrices;

/**
* @class StockPredictor
* @brief Manages the predition for future stock values after loading a time series and trained model for desired stock.
* The front end browser can
* <ul>
* <li> Trained Model Visualization
* <li> Validation Dataset Visualization
* <li> Future Price Data Visualization
* </ul>
*/
class StockPredictor {
public:
  /**
  * @brief Default Constructor
  */
  StockPredictor();
  /** 
  * @brief Method to load the already trained neural Network LSTM Model
  * @param [in] stockSymbol The BSE stock symbol use for prediction.
  */
  void loadModel(const std::string &stockSymbol);
  /** 
  * @brief Method to make N days future predictions
  * @param [in] N The future number of days.
  */
  void predict(const int64_t N);
  /**
  * @brief Method to initiate the visualization of model with given arguments
  * @param [in] args The arguments received from the front-end web browser client.
  */
  void testModel(const std::string &args);
  /** 
  * @brief Default Destructor
  */
  virtual ~StockPredictor();

private:
  /**
  * @brief Helper Method to load historical stock prices for stock
  */
  void loadTimeSeries();
  /**
  * @brief Helper Method to perform prediction for given input set and desired output
  * @param [in] input The normalized input vector for prediction.
  * @param [in] expectedOuput <i>optional</i> The normalized expected output vector (used with validation dataset).
  */
  std::vector<float> predict(const std::vector<float> &input,
                             const std::vector<float> &expectedOuput = {});
  /**
  * @brief Helper Method to log the results into a csv file
  * @param [in] logFileName The log file name used for writing the predicted result.
  * @param [in] y_test The expected data vector written to the file.
  * @param [in] input <i>optional</i> The trained data vector written to the file.
  * @param [in] allDates <i>optional</i> The date vector for which the predictions were made. Not required in case for future dates predictions.
  */
  void fileLogger(const std::string &logFileName,
                  const std::vector<float> &y_test,
                  const std::vector<float> &input = {},
                  const std::vector<std::string> &allDates = {}) const;
  MinMaxScaler<float> m_minmaxScaler;            ///< The MinMaxScaler object for (de)normalizing data
  std::shared_ptr<StockLSTM> m_lstmNetwork;      ///< The pointer to StockLSTM neural network object
  std::shared_ptr<StockPrices> m_stockPrices;    ///< The pointer to StockPrices for processing raw stock values
  std::string m_stockSymbol;                     ///< The BSE stock symbol
};

#endif //! STOCKPREDICTOR_HPP_