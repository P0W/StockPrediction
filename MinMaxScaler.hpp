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

/**
* @class MinMaxScaler
* @brief A class responsible for normalizing and de-normalizing given raw stock values
*/
template <typename T> class MinMaxScaler {
public:
 /**
 * @brief Default Constructor 
 */
  MinMaxScaler();
  /**
  * @brief Default Destructor
  */
  ~MinMaxScaler();
  /**
  * @brief Deleted Copy Constructor
  */
  MinMaxScaler(const MinMaxScaler &) = delete;
  /**
  * @brief Deleted Copy Assignment Operator
  */
  MinMaxScaler &operator=(const MinMaxScaler &) = delete;
  /**
  * @brief Method to fit a given data set in range [0,1].
  * @param [in] rawData The vector of raw values (not normalized).
  */
  std::vector<T> fit_transform(const std::vector<T> &rawData);
  /**
  * @brief Method denormalize a given dataset.
  * @param [in] normalizedData The vector of normalized values.
  */
  std::vector<T> inverse(const std::vector<T> &normalizedData) const;
  /**
  * @brief Overloaded parenthesis operator to implicitly perform inverse operation
  * @param [in] normalizedVal The normalized value whose raw value is required.
  */
  T operator()(const T &normalizedVal) const;
  /**
  * @brief Method to perfor normalization for a single raw value
  * @param [in] rawValue The raw value whose normalized value is required.
  */
  T transform(const T &rawValue) const;

private:
  T m_minVal;///< The minimum value in the dataset
  T m_maxVal;///< The maximum value in the dataset
};

template <typename T>
MinMaxScaler<T>::MinMaxScaler() : m_minVal(), m_maxVal() {}

template <typename T> MinMaxScaler<T>::~MinMaxScaler() {}

template <typename T>
std::vector<T> MinMaxScaler<T>::fit_transform(const std::vector<T> &rawData) {
  std::vector<T> result;

  auto minMax = std::minmax_element(std::begin(rawData), std::end(rawData));
  m_minVal = *minMax.first;
  m_maxVal = *minMax.second;

  const T denominator = *minMax.second - *minMax.first;
  const T min = *minMax.first;

  result.reserve(rawData.size());
  std::transform(std::begin(rawData), std::end(rawData),
                 std::back_inserter(result),
                 [min, denominator](const float &price) {
                   return (price - min) / denominator;
                 });

  return result;
}

template <typename T>
std::vector<T>
MinMaxScaler<T>::inverse(const std::vector<T> &normalizedData) const {
  std::vector<T> result;

  const T denominator = m_maxVal - m_minVal;
  const T minVal = m_minVal;
  result.reserve(normalizedData.size());
  std::transform(std::begin(normalizedData), std::end(normalizedData),
                 std::back_inserter(result),
                 [denominator, minVal](const float &price) {
                   return (price * denominator) + minVal;
                 });

  return result;
}

template <typename T>
T MinMaxScaler<T>::operator()(const T &normalizedVal) const {
  const T denominator = m_maxVal - m_minVal;
  return (normalizedVal * denominator) + m_minVal;
}
template <typename T>
inline T MinMaxScaler<T>::transform(const T &rawValue) const {
  const T denominator = m_maxVal - m_minVal;
  return (rawValue - m_minVal) / denominator;
}
#endif /* MINMAXSCALER_HPP_ */
