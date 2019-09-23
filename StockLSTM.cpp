/*
 * StockLSTM.cpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#include "StockLSTM.hpp"

NaiveLSTM::NaiveLSTM(int input_sz, int hidden_sz)
    : input_sz(input_sz), hidden_sz(hidden_sz) {
  W_ii = register_parameter("W_ii", torch::randn({input_sz, hidden_sz}));
  W_hi = register_parameter("W_hi", torch::randn({hidden_sz, hidden_sz}));
  b_i = register_parameter("b_i", torch::randn(hidden_sz));

  W_if = register_parameter("W_if", torch::randn({input_sz, hidden_sz}));
  W_hf = register_parameter("W_hf", torch::randn({hidden_sz, hidden_sz}));
  b_f = register_parameter("b_f", torch::randn(hidden_sz));

  W_ig = register_parameter("W_ig", torch::randn({input_sz, hidden_sz}));
  W_hg = register_parameter("W_hg", torch::randn({hidden_sz, hidden_sz}));
  b_g = register_parameter("b_g", torch::randn(hidden_sz));

  W_io = register_parameter("W_io", torch::randn({input_sz, hidden_sz}));
  W_ho = register_parameter("W_ho", torch::randn({hidden_sz, hidden_sz}));
  b_o = register_parameter("b_o", torch::randn(hidden_sz));
}

torch::Tensor NaiveLSTM::forward(const torch::Tensor& x) {
  const auto& seq_sz = x.size(1);
  std::vector<torch::Tensor> hidden_seq;

  auto h_t = torch::zeros(this->hidden_sz).to(x.device());
  auto c_t = torch::zeros(this->hidden_sz).to(x.device());
  torch::Tensor x_t, r1, r2, r3;
  torch::Tensor i_t, f_t, g_t, o_t;
  for (int64_t t = 0; t < seq_sz; ++t) {
    x_t = x.select(Dim::seq, t);

    r1 = torch::matmul(x_t, this->W_ii);
    r2 = torch::matmul(h_t, this->W_hi);
    r3 = r1 + r2 + this->b_i;
    i_t = torch::sigmoid(r3);

    r1 = torch::matmul(x_t, this->W_if);
    r2 = torch::matmul(h_t, this->W_hf);
    r3 = r1 + r2 + this->b_f;
    f_t = torch::sigmoid(r3);

    r1 = torch::matmul(x_t, this->W_ig);
    r2 = torch::matmul(h_t, this->W_hg);
    r3 = r1 + r2 + this->b_g;
    g_t = torch::tanh(r3);

    r1 = torch::matmul(x_t, this->W_io);
    r2 = torch::matmul(h_t, this->W_ho);
    r3 = r1 + r2 + this->b_o;
    o_t = torch::sigmoid(r3);

    c_t = f_t * c_t + i_t * g_t;
    h_t = o_t * torch::tanh(c_t);
    hidden_seq.push_back(h_t.unsqueeze(Dim::batch));
  }
  return torch::cat(hidden_seq, Dim::batch)
      .transpose(Dim::batch, Dim::seq)
      .contiguous();
}

OptimizedLSTM::OptimizedLSTM(int64_t input_sz, int64_t hidden_sz)
    : Module(),
      m_inputSize(input_sz),
      m_hiddenSize(hidden_sz),
      m_weight_ih(register_parameter("m_weight_ih",
                                     torch::randn({input_sz, hidden_sz * 4}))),
      m_weight_hh(register_parameter("m_weight_hh",
                                     torch::randn({hidden_sz, hidden_sz * 4}))),
      m_bias(register_parameter("m_bias", torch::randn(hidden_sz * 4))) {}

OptimizedLSTM::~OptimizedLSTM() {}

torch::Tensor OptimizedLSTM::forward(const torch::Tensor& input) {
  std::vector<torch::Tensor> hidden_seq;

  torch::Tensor h_t, c_t, i_t, f_t, g_t, o_t;
  torch::Tensor input_t, r1, r2, gates{};
  torch::Tensor hidden_seq_tensor;

  h_t = torch::zeros(this->m_hiddenSize).to(input.device());
  c_t = torch::zeros(this->m_hiddenSize).to(input.device());

  int64_t seq_sz = input.size(1);
  for (int64_t t = 0; t < seq_sz; ++t) {
    input_t = input.select(1, t);
    r1 = torch::matmul(input_t, this->m_weight_ih);
    r2 = torch::matmul(h_t, this->m_weight_hh);
    gates = torch::add(r1, r2, 0);
    gates = torch::add(gates, this->m_bias, 0);
    std::cout << "input.sizes()       =" << input.sizes() << std::endl;

    std::cout << "input_t.sizes()       =" << input_t.sizes() << std::endl;

    std::cout << "gates.sizes()       =" << gates.sizes() << std::endl;

    i_t = torch::sigmoid(gates.slice(1, 0, this->m_hiddenSize));  // input
    f_t = torch::sigmoid(gates.slice(1, this->m_hiddenSize,
                                     this->m_hiddenSize * 2));  // # forget
    g_t = torch::tanh(
        gates.slice(1, this->m_hiddenSize * 2, this->m_hiddenSize * 3));
    o_t = torch::sigmoid(gates.slice(1, this->m_hiddenSize * 3));  // output

    c_t = f_t * c_t + i_t * g_t;
    h_t = o_t * torch::tanh(c_t);

    hidden_seq.push_back(h_t.unsqueeze(0));
  }

  hidden_seq_tensor = torch::cat(hidden_seq, 0);

  // reshape from shape (sequence, batch, feature) to (batch, sequence,
  // feature);

  return hidden_seq_tensor.transpose(0, 1).contiguous();
}

StockLSTM::StockLSTM(const torch::nn::LSTMOptions& lstmOpts1,
                     const torch::nn::LSTMOptions& lstmOpts2,
                     const torch::nn::LinearOptions& linearOpts)
    : torch::nn::Module(),
      lstm1(register_module("lstm1", torch::nn::LSTM(lstmOpts1))),
      lstm2(register_module("lstm2", torch::nn::LSTM(lstmOpts2))),
      linear(register_module("linear", torch::nn::Linear(linearOpts))) {}

StockLSTM::~StockLSTM() {}

torch::Tensor StockLSTM::forward(const torch::Tensor& input) {
  torch::nn::RNNOutput lstm_out = this->lstm1->forward(input);

  // std::cout << lstm_out.output.sizes() << '\n';  //[5, 3616, 64]
  lstm_out = this->lstm2->forward(lstm_out.output);

  /*std::cout << lstm_out.output.sizes() << '\n';  // same as above [5, 3616,
  64] std::cout << lstm_out.output[-1].sizes() << '\n';  //[ 3616, 64 ]
  std::cout << lstm_out.output[-1].view({this->batch_size, -1}).sizes()
            << '\n';  //[ 3616, 64 ] */

  torch::Tensor y_pred = this->linear(lstm_out.output[-1]);
  /*std::cout << y_pred.sizes() << '\n';             //[3616,1]
  std::cout << y_pred.view({-1}).sizes() << '\n';  //[3616]*/

  return y_pred.view({-1});
}
