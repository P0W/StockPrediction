/*
 * NetworkConstants.cpp
 *
 *  Created on: 17-Oct-2019
 *      Author: Prashant Srivastava
 */

#ifndef REQUEST_HANDLER_HPP_
#define REQUEST_HANDLER_HPP_

#include <boost/asio.hpp>
#include <memory>

#include "StockPredictor.hpp"

/**
 * @class RequestHandler
 * @brief A class responsible to handle web client requests
 */
class RequestHandler {
public:
  /**
   * @brief Default Constructor
   */
  RequestHandler();
  /**
   * @brief Default Destructor
   */
  virtual ~RequestHandler();
  /**
   * @brief Method used for starting service for handling request
   * @param [in] stockPredictor The pointer to StockPredictor object.
   */
  void setupService(const std::shared_ptr<StockPredictor> &stockPredictor);
  /**
   * @brief Method used for starting listener and receiving request in a thread
   */
  void run();

private:
  boost::asio::io_context m_ioc; ///< The Boost Async IO context object
};

#endif //! REQUEST_HANDLER_HPP_