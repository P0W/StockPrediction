/*
 * Timer.cpp
 *
 *  Created on: 21-Sep-2019
 *      Author: Prashant Srivastava
 */

#include "Timer.hpp"
#include <cstdio>

Timer::Timer(const char* fmtString)
    : startTime(std::chrono::steady_clock::now()),
      endTime(std::chrono::steady_clock::now()),
      fmtString(fmtString) {}

Timer::~Timer() { show(); }

void Timer::show(bool resetStartTime) {
  endTime = std::chrono::steady_clock::now();
  auto elapsedTime =
      std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime)
          .count();
  const double durationInSec = elapsedTime / 1e6;

  if (resetStartTime) {
    startTime = std::chrono::steady_clock::now();
  }

  std::printf(fmtString, durationInSec);
}
