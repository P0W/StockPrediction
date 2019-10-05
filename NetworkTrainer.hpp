/*
 * NetworkTrainer.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef NETWORKTRAINER_HPP_
#define NETWORKTRAINER_HPP_

#include "StockLSTM.hpp"
#include <fstream>
#include <memory>
#include <torch/torch.h>

class NetworkTrainer {
public:
  const float kRunningLoss = 0.0001;
  NetworkTrainer(int64_t input, int64_t hidden, int64_t output,
                 int64_t numLayers, int64_t prevSamples, double learningRate,
                 int64_t maxEpochs, const std::string &modelName,
                 const std::string &companyName);
  virtual ~NetworkTrainer();

  torch::Tensor fit(const torch::Tensor &x_train, const torch::Tensor &y_train);

  void saveModel(const std::string &fileName) const;
  void loadModel(const std::string &fileName);

  virtual void dataWriter(const std::string &fileName,
                          const torch::Tensor &data) {
    (void)data;
    (void)fileName;
  }

private:
  bool gpuAvailable;
  int64_t maxEpochs;
  int64_t prevSamples;
  std::string modelName;
  std::string companyName;
  std::shared_ptr<StockLSTM> lstmNetwork;
  std::shared_ptr<torch::optim::Optimizer> optimizer;
};

#endif /* NETWORKTRAINER_HPP_ */
