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

/**
 * @class Timer
 * @brief A class responsible for knowing time elapsed between two events
 */
class Timer {
public:
  /**
   * @brief Constructor
   * @param [in] fmtString The format in which message needs to be displayed
   * after end of a event.
   */
  Timer(const char *fmtString);
  /**
   * @brief Default Destructor
   */
  virtual ~Timer();

  /**
   * @brief Method to show the elapsed time
   * @param [in] resetStartTime Wether to reset the timer after show
   * <i>optional</i>.
   */
  void show(bool resetStartTime = true);
  /**
   * @brief Overloaded parenthesis operator to return the elapsed time.
   */
  operator double();

private:
  std::chrono::steady_clock::time_point
      startTime; ///< The start time captured at start of event
  std::chrono::steady_clock::time_point
      endTime; ///< The end time captured at end of event

  const char *fmtString;
};

#endif /* TIMER_HPP_ */
