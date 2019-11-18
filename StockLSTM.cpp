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

torch::Tensor NaiveLSTM::forward(const torch::Tensor &x) {
  const auto &seq_sz = x.size(1);
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

StockLSTM::StockLSTM(const torch::nn::LSTMOptions &lstmOpts1,
                     const torch::nn::LSTMOptions &lstmOpts2,
                     const torch::nn::DropoutOptions &dropOutOpts,
                     const torch::nn::LinearOptions &linearOpts)
    : torch::nn::Module(),
      lstm1(register_module("lstm1", torch::nn::LSTM(lstmOpts1))),
      lstm2(register_module("lstm2", torch::nn::LSTM(lstmOpts2))),
      dropOut(register_module("dropOut", torch::nn::Dropout(dropOutOpts))),
      linear(register_module("linear", torch::nn::Linear(linearOpts))) {}

StockLSTM::~StockLSTM() {}

torch::Tensor StockLSTM::forward(const torch::Tensor &input) {
  // std::cout << input.sizes() << '\n';  //[5, 3616, 64]
  // y_pred = sigmoid(y_pred);
  // std::get<0>(
  //  maxTensor)); // 0 is the max values, 1 is the indices of max values
  // std::cout << y_pred.sizes() << '\n';             //[3616,1]
  // std::cout << y_pred.view({-1}).sizes() << '\n';  //[3616]

  // std::cout << lstm_out.output.sizes() << '\n';  //[5, 3616, 64]

  // std::cout << lstm_out.output.sizes() << '\n';  // same as above [5,
  // 3616,64] std::cout << lstm_out.output[-1].sizes() << '\n';  //[ 3616, 64 ]
  // std::cout << lstm_out.output[-1].view({this->batch_size, -1}).sizes() <<
  // '\n';  //[ 3616, 64 ]

  // std::cout << lstm_out.output[-1].sizes() << '\n';
  // std::cout << "temp.sizes()" << std::get<0>(temp).sizes();
  // const auto &maxTensorTuple = torch::max(lstm_out.output, 0);

  // 1. Input tensor is of size (previousSamples, totalBatch, 1) and is
  // feedforward to LSTM Layer -1 All states here are initialized to 0.
  torch::nn::RNNOutput lstm_out = this->lstm1->forward(input);

  // 2. Output of Layer-1 is feedforward to LSTM Layer -2 with the states
  // captures.
  lstm_out = this->lstm2->forward(lstm_out.output, lstm_out.state);

  // 3. Adjust the output tensor which is (totalBatch, 1) and make it to
  // (totalBatch) and feedforward to the Drop Out Layer with probability of 20 %
  auto outTensor = lstm_out.output[-1];
  outTensor = this->dropOut->forward(outTensor);

  // 4. The final output of dropout layer is feedforward on Linear Layer.
  torch::Tensor y_pred = this->linear->forward(outTensor);

  // 5. Linear Layesr out put the the final output for one epoch
  return y_pred.view({-1});
}
