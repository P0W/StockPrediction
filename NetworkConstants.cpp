/*
 * NetworkConstants.cpp
 *
 *  Created on: 22-Sep-2019
 *      Author: Prashant Srivastava
 */

#include "NetworkConstants.hpp"

const int64_t NetworkConstants::kPrevSamples = 3;
const int64_t NetworkConstants::kMaxEpochs = 50000;
const float NetworkConstants::kLearningRate = 1e-3;
const double NetworkConstants::kMinimumLoss = 1e-4;
const float NetworkConstants::kSplitRatio = 0.1;

const int64_t NetworkConstants::input_size = 1;
const int64_t NetworkConstants::hidden_size = 128;
const int64_t NetworkConstants::output_size = 1;
const int64_t NetworkConstants::num_of_layers = 4;

const size_t NetworkConstants::kMaxStockPrices = 6000;

const double NetworkConstants::klsmt1DropOut = 0.0;
const double NetworkConstants::klsmt2DropOut = 0.0;

const std::string NetworkConstants::kRootFolder = "stockData/";
