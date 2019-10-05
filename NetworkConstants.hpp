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

extern const int64_t kPrevSamples;
extern const int64_t kMaxEpochs;
extern const double kMaxTrainTime;
extern const float kLearningRate;
extern const float kSplitRatio;
extern const double kMinimumLoss;
extern const bool kIncludeBias;

extern const int64_t input_size;
extern const int64_t hidden_size;
extern const int64_t output_size;
extern const int64_t num_of_layers;

extern const double klsmt1DropOut;
extern const double klsmt2DropOut;

extern const size_t kMaxStockPrices;

extern const std::string kRootFolder;

} // namespace NetworkConstants

#endif // NETWORKCONSTANTS_HPP_
