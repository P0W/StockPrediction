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

/* Number of previous stock prices to be used to precidict/train the next stock prices */
extern const int64_t kPrevSamples;

/* Maximum number of Epoch, before training can stop, if not converged */
extern const int64_t kMaxEpochs;

/* Maximum amount of time to wait for network to converge */
extern const double kMaxTrainTime;

/* The learning rate for the Optimizer */
extern const float kLearningRate;

/* The percentage of data needed for test/validation data set from entire dataset */
extern const float kSplitRatio;

/* The acceptable minimum loss, in order to stop network training */
extern const double kMinimumLoss;

/* Wether to include bias for LSMT Layer-1 and Layer-2 */
extern const bool kLstmIncludeBias;

/* Wether to include bias for Linear Layer */
extern const bool kIncludeLinearBias;

/* Input Size for LSTM Layer-1 */
extern const int64_t input_size;

/* Hidden Size for LSTM Layer-1 and Input & Hidden size LSTM Layer-2 */
extern const int64_t hidden_size;

/* Input Size for LSTM Layer-1 */
extern const int64_t output_size;

/* Number of units for LSTM Layer-1 and LST Layer-2 */
extern const int64_t num_of_layers;

/* Dropout probability for LSMT Layer-1 */
extern const double klsmt1DropOut;

/* Dropout probability for LSMT Layer-2 */
extern const double klsmt2DropOut;

/* Dropout probability for Drop Out Layer */
extern const double kdropOutDropOut;

/* Maximum number of Stock Prices to consider for training and testing */
extern const size_t kMaxStockPrices;

/* The root folder for saving network data */
extern const std::string kRootFolder;

/* Port at which the client will listen to this server */
extern const unsigned short kClientPort;

} // namespace NetworkConstants

#endif // NETWORKCONSTANTS_HPP_
