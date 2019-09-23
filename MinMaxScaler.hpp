/*
 * MinMaxScaler.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef MINMAXSCALER_HPP_
#define MINMAXSCALER_HPP_

#include <algorithm>
#include <vector>

template <typename T>
class MinMaxScaler {
 public:
  MinMaxScaler();
  ~MinMaxScaler();
  MinMaxScaler(const MinMaxScaler&) = delete;
  MinMaxScaler& operator=(const MinMaxScaler&) = delete;

  std::vector<T> fit_transform(const std::vector<T>& rawData);
  std::vector<T> inverse(const std::vector<T>& normalizedData) const;
  T operator()(const T& normalizedVal) const;

 private:
  T m_minVal;
  T m_maxVal;
};

template <typename T>
MinMaxScaler<T>::MinMaxScaler() : m_minVal(), m_maxVal() {}

template <typename T>
MinMaxScaler<T>::~MinMaxScaler() {}

template <typename T>
std::vector<T> MinMaxScaler<T>::fit_transform(const std::vector<T>& rawData) {
  std::vector<T> result;

  auto minMax = std::minmax_element(std::begin(rawData), std::end(rawData));
  m_minVal = *minMax.first;
  m_maxVal = *minMax.second;

  const T denominator = *minMax.second - *minMax.first;
  const T min = *minMax.first;

  result.reserve(rawData.size());
  std::transform(std::begin(rawData), std::end(rawData),
                 std::back_inserter(result),
                 [min, denominator](const float& price) {
                   return (price - min) / denominator;
                 });

  return result;
}

template <typename T>
std::vector<T> MinMaxScaler<T>::inverse(
    const std::vector<T>& normalizedData) const {
  std::vector<T> result;

  const T denominator = m_maxVal - m_minVal;
  const T minVal = m_minVal;
  result.reserve(normalizedData.size());
  std::transform(std::begin(normalizedData), std::end(normalizedData),
                 std::back_inserter(result),
                 [denominator, minVal](const float& price) {
                   return (price * denominator) + minVal;
                 });

  return result;
}

template <typename T>
T MinMaxScaler<T>::operator()(const T& normalizedVal) const {
  const T denominator = m_maxVal - m_minVal;
  return (normalizedVal * denominator) + m_minVal;
}
#endif /* MINMAXSCALER_HPP_ */
