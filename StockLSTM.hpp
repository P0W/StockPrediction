/*
 * StockLSTM.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef STOCK_LSTM_HPP_
#define STOCK_LSTM_HPP_

#include <torch/torch.h>

/**
* @class NaiveLSTM
* @brief A class representing a single LSTM module
*/
class NaiveLSTM : public torch::nn::Module {
 /**
 * @brief Constructor
 * @param [in] input_sz The input size for the LSTM.
 * @param [in] hidden_sz The hidden size for the LSTM.
 */
  NaiveLSTM(int input_sz, int hidden_sz);
 /**
 * @brief Method for feed-forwarding the input tensor to the network
 * @param[in] input The input Pytorch Tensor.
 */
  torch::Tensor forward(const torch::Tensor &input);
  /**
  * @enum Dim
  * @brief Enumeration to represent batch, sequence and feature part of Pytorch input tensor data.
  */
  enum Dim : int64_t { batch = 0, seq = 1, feature = 2 };

private:
  int input_sz;        ///< The input size for LSTM
  int hidden_sz;       ///< The hidden size for LSTM
                    
  torch::Tensor W_ii;  ///< The weight for first input layer
  torch::Tensor W_hi;  ///< The weight for input hidden layer
  torch::Tensor b_i;   ///< The weight for biases for input layer

  torch::Tensor W_if;  ///< The weight for forget input layer
  torch::Tensor W_hf;  ///< The weight for forget hidden layer
  torch::Tensor b_f;   ///< The weight for biases for forget layer 

  torch::Tensor W_ig;  ///< The weight for control input layer
  torch::Tensor W_hg;  ///< The weight for control hidden layer
  torch::Tensor b_g;   ///< The weight for biases for control layer 

  torch::Tensor W_io;  ///< The weight for output input layer
  torch::Tensor W_ho;  ///< The weight for output hidden layer
  torch::Tensor b_o;   ///< The weight for biases for ouput layer 
};

/**
* @class StockLSTM
* @brief A class abstracting the Pytorch's LSTM modules and reprsenting the entire neural network
*/
class StockLSTM : public torch::nn::Module {
public:
  /**
  * @brief Constructor
  * @param [in] lstmOpts1 The reference to Pytorch LSTMOptions for Layer-1.
  * @param [in] lstmOpts2 The reference to Pytorch LSTMOptions for Layer-2.
  * @param [in] dropOutOpts The reference to Pytorch DropoutOptions for the drop out layer.
  * @param [in] linearOpts The reference to Pytorch LinearOptions for the final linear layer.
  */
  StockLSTM(const torch::nn::LSTMOptions &lstmOpts1,
            const torch::nn::LSTMOptions &lstmOpts2,
            const torch::nn::DropoutOptions& dropOutOpts,
            const torch::nn::LinearOptions &linearOpts);
  /**
  * @brief Default Destructor
  */
  ~StockLSTM();
  /**
  * @brief The overloaded forward method from torch::nn::Module
  * @param [in] input The input Pytorch Tensor
  */
  torch::Tensor forward(const torch::Tensor &input);

private:
  torch::nn::LSTM lstm1;      ///< The LSTM Layer-1 object
  torch::nn::LSTM lstm2;      ///< The LSTM Layer-2 object
  torch::nn::Dropout dropOut; ///< The DropOut Layer object
  torch::nn::Linear linear;   ///< The Linear layer object
};

#endif // !STOCK_LSTM_HPP_
