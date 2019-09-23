/*
 * ITimeSeries.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef ITIMESERIES_HPP_
#define ITIMESERIES_HPP_

class ITimeSeries {
 public:
  virtual ~ITimeSeries() {}
  virtual bool loadTimeSeries() = 0;
  virtual void normalizeData() = 0;
  virtual void reshapeSeries(float testSplitRatio) = 0;
};

#endif /* ITIMESERIES_HPP_ */
