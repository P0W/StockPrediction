/*
 * ITimeSeries.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef ITIMESERIES_HPP_
#define ITIMESERIES_HPP_

/**
* @class ITimeSeries
* @brief An interface used to putting following functionalities
* <ul>
* <li> Loading Time Series Data.
* <li> Normalizing Time Series Data.
* <li> Splitting and Reshaping Time Series Data into Training and Validation datasets.
* </ul>
*/
class ITimeSeries {
public:
  /**
  * @brief Default Destructor
  */
  virtual ~ITimeSeries() {}
  /**
  * @brief Abstract method utilized for loading time series data
  */
  virtual bool loadTimeSeries(const std::string &, const int64_t&) = 0;
  /**
  * @brief Abstract method utilized for normalizing time series data
  */
  virtual void normalizeData() = 0;
  /**
  * @brief Abstract method utilized for splitting and reshaping time series data
  */
  virtual void reshapeSeries(float testSplitRatio, int64_t) = 0;
};

#endif /* ITIMESERIES_HPP_ */
