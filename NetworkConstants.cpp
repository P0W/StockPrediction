/*
 * NetworkConstants.cpp
 *
 *  Created on: 22-Sep-2019
 *      Author: Prashant Srivastava
 */

#include "NetworkConstants.hpp"

const int64_t NetworkConstants::kPrevSamples = 5;
const int64_t NetworkConstants::kMaxEpochs = 1e4;
const double NetworkConstants::kMaxTrainTime = 900.0; // seconds
const float NetworkConstants::kLearningRate = 0.005f;
const double NetworkConstants::kMinimumLoss = 4e-4;
const float NetworkConstants::kSplitRatio = 0.01;
const bool NetworkConstants::kLstmIncludeBias = false;
const bool NetworkConstants::kIncludeLinearBias = false;

const int64_t NetworkConstants::input_size = 1;
const int64_t NetworkConstants::hidden_size = 32;
const int64_t NetworkConstants::output_size = 1;
const int64_t NetworkConstants::num_of_layers = 4;

const size_t NetworkConstants::kMaxStockPrices = 5000;

const double NetworkConstants::klsmt1DropOut = 0.1;
const double NetworkConstants::klsmt2DropOut = 0.2;
const double NetworkConstants::kdropOutDropOut = 0.2;

const std::string NetworkConstants::kRootFolder = "stockData/";

const unsigned short NetworkConstants::kClientPort = 4242;
