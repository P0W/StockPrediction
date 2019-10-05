/*
 * Timer.hpp
 *
 *  Created on: 21-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>
#include <string>

class Timer {
public:
  Timer(const char *fmtString);
  virtual ~Timer();

  void show(bool resetStartTime = true);
  operator double();

private:
  std::chrono::steady_clock::time_point startTime;
  std::chrono::steady_clock::time_point endTime;

  const char *fmtString;
};

#endif /* TIMER_HPP_ */
