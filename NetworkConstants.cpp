/*
 * NetworkConstants.cpp
 *
 *  Created on: 22-Sep-2019
 *      Author: Prashant Srivastava
 */

#include "NetworkConstants.hpp"

const int64_t NetworkConstants::kPrevSamples = 5;
const int64_t NetworkConstants::kMaxEpochs = 50000;
const float NetworkConstants::kLearningRate = 1e-2;
const double NetworkConstants::kMinimumLoss = 4e-4;
const float NetworkConstants::kSplitRatio = 0.01;
const bool NetworkConstants::kIncludeBias = false;

const int64_t NetworkConstants::input_size = 1;
const int64_t NetworkConstants::hidden_size = 64;
const int64_t NetworkConstants::output_size = 1;
const int64_t NetworkConstants::num_of_layers = 4;

const size_t NetworkConstants::kMaxStockPrices = 6000;

const double NetworkConstants::klsmt1DropOut = 0.0;
const double NetworkConstants::klsmt2DropOut = 0.0;

const std::string NetworkConstants::kRootFolder = "stockData/";
