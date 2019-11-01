/*
 * StockLSTM.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef STOCK_LSTM_HPP_
#define STOCK_LSTM_HPP_

#include <torch/torch.h>

// Define a new Module.
struct NaiveLSTM : public torch::nn::Module {
  NaiveLSTM(int input_sz, int hidden_sz);
  torch::Tensor forward(const torch::Tensor &x);

  enum Dim : int64_t { batch = 0, seq = 1, feature = 2 };

private:
  int input_sz;
  int hidden_sz;

  torch::Tensor W_ii;
  torch::Tensor W_hi;
  torch::Tensor b_i;

  torch::Tensor W_if;
  torch::Tensor W_hf;
  torch::Tensor b_f;

  torch::Tensor W_ig;
  torch::Tensor W_hg;
  torch::Tensor b_g;

  torch::Tensor W_io;
  torch::Tensor W_ho;
  torch::Tensor b_o;
};

class OptimizedLSTM : public torch::nn::Module {
public:
  OptimizedLSTM(int64_t input_sz, int64_t hidden_sz);
  ~OptimizedLSTM();
  torch::Tensor forward(const torch::Tensor &input);

private:
  int64_t m_inputSize;
  int64_t m_hiddenSize;
  torch::Tensor m_weight_ih;
  torch::Tensor m_weight_hh;
  torch::Tensor m_bias;
};

class StockLSTM : public torch::nn::Module {
public:
  StockLSTM(const torch::nn::LSTMOptions &lstmOpts1,
            const torch::nn::LSTMOptions &lstmOpts2,
            const torch::nn::DropoutOptions& dropOutOpts,
            const torch::nn::LinearOptions &linearOpts);
  ~StockLSTM();

  torch::Tensor forward(const torch::Tensor &x);

private:
  torch::nn::LSTM lstm1;
  torch::nn::LSTM lstm2;
  torch::nn::Dropout dropOut;
  torch::nn::Linear linear;
};

#endif // !STOCK_LSTM_HPP_
