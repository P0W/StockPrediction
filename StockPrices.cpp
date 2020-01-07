/*
 * StockPrices.cpp
 *
 *  Created on: 19-Sep-2019
 *      Author: Prashant Srivastava
 */

#include "csv.h"
#ifdef _MSC_VER
#include <Windows.h>
#pragma comment(lib, "urlmon.lib")
#else
#include <curl/curl.h>
#endif

#include "MinMaxScaler.hpp"
#include "StockPrices.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

size_t downloadDataCallback(void *ptr, size_t size, size_t nmemb,
                            FILE *stream) {
  size_t written = fwrite(ptr, size, nmemb, stream);

  std::printf("Downloaded %zd byte(s).\n", written);
  return written;
}

const char *quandlToken = "?api_key=p0wB3L7ZHSVh3sp0wV5cyRZPFp0W";

const char *url = "https://www.quandl.com/api/v3/datasets/BSE/";

int downloadStockData(const std::string &stockSymbol) {
  char csvFileName[FILENAME_MAX];
  char queryUrl[FILENAME_MAX];

  strcpy(csvFileName, stockSymbol.c_str());
  strcat(csvFileName, ".csv");

  std::string downLoadFile = NetworkConstants::kRootFolder + csvFileName;

  strcpy(queryUrl, url);
  strcat(queryUrl, csvFileName);
  std::printf("Fetching %s [%s] ....\n", downLoadFile.c_str(), queryUrl);
  strcat(queryUrl, quandlToken);

#ifndef _WIN32
  FILE *fp = nullptr;
  CURL *curl = curl_easy_init();
  if (curl) {
    fp = fopen(downLoadFile.c_str(), "wb");
    curl_easy_setopt(curl, CURLOPT_URL, queryUrl);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, downloadDataCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    (void)curl_easy_perform(curl);

    curl_easy_cleanup(curl);
    fclose(fp);
    return 0;
  }
#elif defined(_MSC_VER)

  HRESULT res =
      URLDownloadToFile(NULL, queryUrl, downLoadFile.c_str(), 0, NULL);
  if (res == S_OK) {
    std::printf("Ok\n");
    return 0;
  } else if (res == E_OUTOFMEMORY) {
    std::printf("Buffer length invalid, or insufficient memory\n");
  } else if (res == INET_E_DOWNLOAD_FAILURE) {
    std::printf("URL is invalid\n");
  } else {
    std::printf("Other error: %d\n", res);
  }
#endif
  return -1;
}
} // namespace

StockData::StockData(const std::string &date, float val)
    : m_date(date), m_closePrice(val), m_dateNum(0), m_hasError(false) {

  int year, month, day;
  if (std::sscanf(m_date.c_str(), "%d-%d-%d", &year, &month, &day) == 3) {
    m_dateNum = 10000 * year + 100 * month + day;
    m_hasError = false;
  } else {
    m_hasError = true;
  }
}

bool StockData::operator<(const StockData &other) {
  return other.m_dateNum > this->m_dateNum;
}

float StockData::getClosePrice() const { return m_closePrice; }
std::string StockData::getDate() const { return m_date; }

bool StockData::wasBadEntry() const { return m_hasError; }

StockPrices::StockPrices(MinMaxScaler<float> &scaler)
    : scaler(scaler), stockClosePrices{},
      normalizedStockClosePrices{}, dates{}, x_train{}, y_train{} {}

StockPrices::~StockPrices() {}

bool StockPrices::loadTimeSeries(const std::string &stockSymbol,
                                 const size_t &maxDataSize) {
#ifdef WIN32
  const auto pos = stockSymbol.find_last_of('\\');
#else
  const auto pos = stockSymbol.find_last_of('/');
#endif
  std::string rawStockSymbol = stockSymbol;

  if (pos != std::string::npos) {
    rawStockSymbol = stockSymbol.substr(pos + 1);
  }

  std::string stockValuesDownloadedFile =
      NetworkConstants::kRootFolder + rawStockSymbol + ".csv";

  // Check if file exists
  std::ifstream fileHandle(stockValuesDownloadedFile);
  if (!fileHandle.good()) {
    std::printf("Downloading %s ...\n", rawStockSymbol.c_str());
    int errorCode = downloadStockData(rawStockSymbol);
    if (errorCode != 0) {
      std::printf("Cannot load %s\n", rawStockSymbol.c_str());
      return true;
    }
  }
  fileHandle.close();
  io::CSVReader<2> in(stockValuesDownloadedFile);
  try {
    in.read_header(io::ignore_extra_column, "Date", "Close");
  } catch (std::exception e) {
    const auto errorMsg = e.what();
    std::printf("%s - error reading %s\n", errorMsg, rawStockSymbol.c_str());
    return true;
  }

  std::string date;
  float close;

  std::vector<StockData> stockData;

  while (in.read_row(date, close)) {
    stockData.emplace_back(StockData{date, close});
  }

  if (std::any_of(std::begin(stockData), std::begin(stockData),
                  [](const StockData &s) { return s.wasBadEntry(); })) {
    return true;
  }

  std::sort(std::begin(stockData), std::end(stockData));

  size_t vecSize = stockData.size();
  // Grab only maxDataSize stock prices
  if (vecSize > maxDataSize) {
    stockData.erase(std::begin(stockData),
                    std::begin(stockData) + (vecSize - maxDataSize));
  }
  stockClosePrices.reserve(stockData.size());
  dates.reserve(stockClosePrices.size());

  std::transform(std::begin(stockData), std::end(stockData),
                 std::back_inserter(stockClosePrices),
                 [](const StockData &s) { return s.getClosePrice(); });

  std::transform(std::begin(stockData), std::end(stockData),
                 std::back_inserter(dates),
                 [](const StockData &s) { return s.getDate(); });

  return false;
}
void StockPrices::reshapeSeries(float testSplitRatio,
                                int64_t num_prev_samples) {
  int64_t dataSet_size = normalizedStockClosePrices.size();
  int64_t testSize = static_cast<int64_t>(dataSet_size * testSplitRatio);
  int64_t trainSize = dataSet_size - testSize;

  int64_t idx, xIdx;

  for (idx = num_prev_samples; idx < trainSize; ++idx) {
    for (xIdx = idx - num_prev_samples; xIdx < idx; ++xIdx) {
      x_train.push_back(normalizedStockClosePrices[xIdx]);
    }

    y_train.push_back(normalizedStockClosePrices[idx]);
  }

  // Adjust dates, strip off first num_prev_samples
  dates_train.reserve(y_train.size());
  // dates.erase(std::begin(dates), std::begin(dates) + num_prev_samples);
  std::copy(std::cbegin(dates) + num_prev_samples,
            std::cbegin(dates) + num_prev_samples + y_train.size(),
            std::back_inserter(dates_train));

  assert(dates_train.size() == y_train.size());

  auto startIndx = trainSize - num_prev_samples;
  // Test Data Set
  for (idx = num_prev_samples; idx < testSize + num_prev_samples; ++idx) {
    for (xIdx = idx - num_prev_samples; xIdx < idx; ++xIdx) {
      x_test.push_back(normalizedStockClosePrices[startIndx + xIdx]);
    }

    y_test.push_back(normalizedStockClosePrices[startIndx + idx]);
  }

  // Adjust dates, strip off first num_prev_samples
  dates_test.reserve(y_test.size());
  std::copy(std::cbegin(dates) + startIndx + num_prev_samples, std::cend(dates),
            std::back_inserter(dates_test));

  assert(testSize == static_cast<int64_t>(y_test.size()));
  assert(dates_test.size() == y_test.size());
  assert(normalizedStockClosePrices.size() - num_prev_samples ==
         y_test.size() + y_train.size());
}

void StockPrices::normalizeData() {
  normalizedStockClosePrices = scaler.fit_transform(stockClosePrices);
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<std::string>>
StockPrices::getTrainData() const {
  return std::make_tuple(x_train, y_train, dates_train);
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<std::string>>
StockPrices::getTestData() const {
  return std::make_tuple(x_test, y_test, dates_test);
}
