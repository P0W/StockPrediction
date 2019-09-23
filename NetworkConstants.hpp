/*
 * NetworkConstants.hpp
 *
 *  Created on: 22-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef NETWORKCONSTANTS_HPP_
#define NETWORKCONSTANTS_HPP_

#include <cstdint>
#include <string>

namespace NetworkConstants {

extern const int64_t prevSamples;
extern const int64_t maxEpochs;
extern const float learningRate;
extern const float splitRatio;
extern const double kMinimumLoss;

extern const int64_t input_size;
extern const int64_t hidden_size;
extern const int64_t output_size;
extern const int64_t num_of_layers;

extern const double klsmt1DropOut;
extern const double klsmt2DropOut;

extern const size_t kMaxStockPrices;

extern const std::string rootFolder;

}  // namespace NetworkConstants

#endif  // NETWORKCONSTANTS_HPP_
