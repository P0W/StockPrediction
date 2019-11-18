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

/**
 * Forward declaration for classes
 */
class StockLSTM;
namespace torch {
namespace optim {
class Optimizer;
}
} // namespace torch

/**
 * @class NetworkTrainer
 * @brief A class responsible for
 * <ul>
 * <li> Training a StockLSTM Network.
 * <li> Saving a trained StockLSTM Network.
 * <li> Loading a already trained StockLSTM Network.
 * <li> Provding logging rotinues to inspect a StockLSTM Network.
 * </ul>
 */
class NetworkTrainer {
public:
  const float kRunningLoss = 0.0001; ///< The global desired minimum loss
  /**
   * @brief Constructor
   * @param [in] modelName The symbol of the stock is the neural network model
   * name.
   * @param [in] companyName The name of the company as listed on BSE.
   */
  NetworkTrainer(const std::string &modelName, const std::string &companyName);
  /**
   * @brief Default Destructor
   */
  virtual ~NetworkTrainer();
  /**
   * @brief Method to perform training for neural network.
   * @param [in] x_train The normalized training input dataset.
   * @param [in] y_train The normalized training expected output dataset.
   * @param [in] x_test The normalized validation input dataset.
   * @param [in] y_test The normalized validation expected output dataset.
   */
  std::vector<float> fit(const std::vector<float> &x_train,
                         const std::vector<float> &y_train,
                         const std::vector<float> &x_test = {},
                         const std::vector<float> &y_test = {});
  /**
   * @brief Method to save the trained neural network model.
   * @param [in] fileName The file name used for saving the neural network.
   */
  void saveModel(const std::string &fileName) const;
  /**
   * @brief Method to load an already trained neural network model.
   * @param [in] fileName The file name used for loading the neural network.
   */
  void loadModel(const std::string &fileName);

  /**
   * @brief A callback method to customize writing to a file, the trained neural
   * network output data.
   * @param [in] fileName The file name on which data needs to be written.
   * @param [in] data The neural network data that will be written to the given
   * file.
   */
  virtual void dataWriter(const std::string &fileName,
                          const std::vector<float> &data) {
    /* Dummy empty implementation */
    (void)data;
    (void)fileName;
  }

private:
  bool gpuAvailable; ///< Variable to know if GPU is available on given system
  std::string modelName; ///< The name given to the neural network model
  std::string
      companyName; ///< The name of the company whose model is been trained
  std::shared_ptr<StockLSTM>
      lstmNetwork; ///< The pointer to StockLSTM neural network object
  std::shared_ptr<torch::optim::Optimizer>
      optimizer; ///< The pointer to Pytorch's Optimizer object
};

#endif /* NETWORKTRAINER_HPP_ */
