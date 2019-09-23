/*
 * StockPrices.cpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#include "StockPrices.hpp"
#include <curl/curl.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include "MinMaxScaler.hpp"
#include "NetworkConstants.hpp"
#include "csv.h"

namespace {

size_t downloadDataCallback(void *ptr, size_t size, size_t nmemb,
                            FILE *stream) {
  size_t written = fwrite(ptr, size, nmemb, stream);

  std::printf("Downloaded %ld byte(s).\n", written);
  return written;
}

const char *quandlToken = "?api_key=B3L7ZHSVh3sV5cyRZPFW";

const char *url = "https://www.quandl.com/api/v3/datasets/BSE/";

int downloadStockData(const std::string &stockSymbol) {
  CURL *curl;
  FILE *fp;

  char csvFileName[FILENAME_MAX];
  char queryUrl[FILENAME_MAX];

  strcpy(csvFileName, stockSymbol.c_str());
  strcat(csvFileName, ".csv");

  std::string downLoadFile = NetworkConstants::rootFolder + csvFileName;

  strcpy(queryUrl, url);
  strcat(queryUrl, csvFileName);
  std::printf("Fectching %s [%s] ....\n", downLoadFile.c_str(), queryUrl);
  strcat(queryUrl, quandlToken);

  curl = curl_easy_init();
  if (curl) {
    fp = fopen(downLoadFile.c_str(), "wb");
    curl_easy_setopt(curl, CURLOPT_URL, queryUrl);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, downloadDataCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    (void)curl_easy_perform(curl);

    curl_easy_cleanup(curl);
    fclose(fp);
  }
  return 0;
}
}  // namespace

StockData::StockData(std::string date, float val)
    : date(date), closePrice(val) {}

bool StockData::operator<(const StockData &other) {
  return other.closePrice > this->closePrice;
}

float StockData::getClosePrice() const { return closePrice; }

StockPrices::StockPrices(const std::string &stockSymbol, int64_t prevSamples,
                         MinMaxScaler<float> &scaler)
    : stockSymbol(stockSymbol),
      num_prev_samples{prevSamples},
      scaler(scaler),
      stockClosePrices{},
      normalizedStockClosePrices{},
      x_train{},
      y_train{} {}

StockPrices::~StockPrices() {}

void StockPrices::loadTimeSeries() {
  (void)downloadStockData(stockSymbol);

  io::CSVReader<2> in(NetworkConstants::rootFolder + stockSymbol + ".csv");
  in.read_header(io::ignore_extra_column, "Date", "Close");
  std::string date;
  float close;

  std::vector<StockData> stockData;

  while (in.read_row(date, close)) {
    stockData.emplace_back(StockData{date, close});
  }

  std::sort(std::begin(stockData), std::end(stockData));

  size_t vecSize = stockData.size();
  // Grab only NetworkConstants::kMaxStockPrices stock prices
  if (vecSize > NetworkConstants::kMaxStockPrices) {
    stockData.erase(
        std::begin(stockData),
        std::begin(stockData) + (vecSize - NetworkConstants::kMaxStockPrices));
  }
  stockClosePrices.reserve(stockData.size());

  std::transform(std::begin(stockData), std::end(stockData),
                 std::back_inserter(stockClosePrices),
                 [](const StockData &s) { return s.getClosePrice(); });
}
void StockPrices::reshapeSeries(float testSplitRatio) {
  int64_t test_size =
      static_cast<int64_t>(normalizedStockClosePrices.size() * testSplitRatio);
  int64_t trainSize = normalizedStockClosePrices.size() - test_size;

  for (int64_t idx = num_prev_samples; idx < trainSize; ++idx) {
    for (int64_t xIdx = idx - num_prev_samples; xIdx < idx; ++xIdx) {
      x_train.push_back(normalizedStockClosePrices[xIdx]);
    }

    y_train.push_back(normalizedStockClosePrices[idx]);
  }
}

void StockPrices::normalizeData() {
  normalizedStockClosePrices = scaler.fit_transform(stockClosePrices);
}

std::pair<std::vector<float>, std::vector<float>> StockPrices::getTrainData()
    const {
  return std::make_pair(x_train, y_train);
}
