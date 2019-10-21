#ifndef REQUEST_HANDLER_HPP_
#define REQUEST_HANDLER_HPP_

#include <boost/asio.hpp>
#include <memory>

#include "StockPredictor.hpp"

class RequestHandler {
public:
  RequestHandler();
  virtual ~RequestHandler();

  void setupService(const std::shared_ptr<StockPredictor> &);

  void run();

private:
  boost::asio::io_context m_ioc;
};

#endif //! REQUEST_HANDLER_HPP_