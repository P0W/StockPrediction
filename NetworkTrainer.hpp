/*
 * NetworkTrainer.hpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#ifndef NETWORKTRAINER_HPP_
#define NETWORKTRAINER_HPP_

#include <memory>
#include <string>
#include <vector>

class StockLSTM;
namespace torch {
namespace optim {
class Optimizer;
}
} // namespace torch
class NetworkTrainer {
public:
  const float kRunningLoss = 0.0001;
  NetworkTrainer(const std::string &modelName,
                 const std::string &companyName);
  virtual ~NetworkTrainer();

  std::vector<float> fit(const std::vector<float> &x_train,
                         const std::vector<float> &y_train,
                         const std::vector<float> &x_test = {},
                         const std::vector<float> &y_test = {});

  void saveModel(const std::string &fileName) const;
  void loadModel(const std::string &fileName);

  virtual void dataWriter(const std::string &fileName,
                          const std::vector<float> &data) {
    (void)data;
    (void)fileName;
  }

private:
  bool gpuAvailable;
  std::string modelName;
  std::string companyName;
  std::shared_ptr<StockLSTM> lstmNetwork;
  std::shared_ptr<torch::optim::Optimizer> optimizer;
};

#endif /* NETWORKTRAINER_HPP_ */
