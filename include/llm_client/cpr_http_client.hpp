#pragma once

#include "llm_client/http_client_interface.hpp"
#include "llm_client/logger.hpp"
#include <algorithm>
#include <chrono>
#include <cpr/cpr.h>
#include <thread>

namespace llm_client {

/**
 * @brief CPR (C++ Requests) 기반의 실제 HTTP 통신 구현체
 */
class CprHttpClient : public IHttpClient {
public:
  CprHttpClient() = default;
  ~CprHttpClient() override = default;

  HttpResponse post(const std::string &url,
                    const std::map<std::string, std::string> &headers,
                    const std::string &body, int timeout_ms = 30000,
                    int max_retries = 0) override {
    cpr::Header cpr_headers;
    for (const auto &[k, v] : headers) {
      cpr_headers.insert({k, v});
    }

    cpr::Session session;
    session.SetUrl(cpr::Url{url});
    session.SetHeader(cpr_headers);
    session.SetBody(cpr::Body{body});
    session.SetTimeout(cpr::Timeout{timeout_ms});

    cpr::Response r;
    int attempts = 0;

    while (true) {
      r = session.Post();

      if (r.error.code == cpr::ErrorCode::OK && r.status_code == 200) {
        break;
      }

      if (isRetryable(r) && attempts < max_retries) {
        attempts++;
        int delay_ms = computeRetryDelayMs(r, attempts);
        LLM_LOG_WARN("HTTP POST to {} failed (HTTP {}, error: {}). Retrying ({}/{}) in {}ms...",
                     url, r.status_code, r.error.message, attempts, max_retries, delay_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        continue;
      }

      break;
    }

    HttpResponse resp;
    resp.status_code = r.status_code;
    resp.text = r.text;
    resp.error_code = static_cast<int>(r.error.code);
    resp.error_message = r.error.message;
    for (const auto &[k, v] : r.header) {
      resp.headers[k] = v;
    }
    return resp;
  }

  HttpResponse postStream(const std::string &url,
                          const std::map<std::string, std::string> &headers,
                          const std::string &body,
                          std::function<void(const std::string &line)> on_line,
                          int timeout_ms = 60000,
                          int max_retries = 0) override {
    cpr::Header cpr_headers;
    for (const auto &[k, v] : headers) {
      cpr_headers.insert({k, v});
    }

    cpr::Session session;
    session.SetUrl(cpr::Url{url});
    session.SetHeader(cpr_headers);
    session.SetBody(cpr::Body{body});
    session.SetTimeout(cpr::Timeout{timeout_ms});

    cpr::Response r;
    int attempts = 0;

    while (true) {
      std::string line_buffer;
      bool any_line_received = false;

      session.SetWriteCallback(cpr::WriteCallback{[&line_buffer, &on_line, &any_line_received](
                                                     std::string_view data, intptr_t) -> bool {
        line_buffer.append(data.data(), data.size());
        size_t pos = 0;
        while ((pos = line_buffer.find('\n')) != std::string::npos) {
          std::string line = line_buffer.substr(0, pos);
          if (!line.empty() && line.back() == '\r') {
            line.pop_back();
          }
          any_line_received = true;
          if (on_line) {
            on_line(line);
          }
          line_buffer.erase(0, pos + 1);
        }
        return true;
      }});

      r = session.Post();

      if (r.error.code == cpr::ErrorCode::OK && r.status_code == 200) {
        if (!line_buffer.empty()) {
          if (line_buffer.back() == '\r') {
            line_buffer.pop_back();
          }
          if (on_line) {
            on_line(line_buffer);
          }
        }
        break;
      }

      bool is_retryable = !any_line_received && isRetryable(r);

      if (any_line_received && r.error.code != cpr::ErrorCode::OK) {
        LLM_LOG_ERROR("Stream interrupted after partial data was delivered; aborting without retry (error: {})",
                      r.error.message);
      }

      if (is_retryable && attempts < max_retries) {
        attempts++;
        int delay_ms = computeRetryDelayMs(r, attempts);
        LLM_LOG_WARN("HTTP stream to {} failed (HTTP {}, error: {}). Retrying ({}/{}) in {}ms...",
                     url, r.status_code, r.error.message, attempts, max_retries, delay_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        continue;
      }

      break;
    }

    HttpResponse resp;
    resp.status_code = r.status_code;
    resp.text = r.text;
    resp.error_code = static_cast<int>(r.error.code);
    resp.error_message = r.error.message;
    for (const auto &[k, v] : r.header) {
      resp.headers[k] = v;
    }
    return resp;
  }

private:
  static bool isRetryable(const cpr::Response &r) {
    return r.error.code != cpr::ErrorCode::OK || r.status_code == 0 ||
           r.status_code == 429 || (r.status_code >= 500 && r.status_code < 600);
  }

  static int computeRetryDelayMs(const cpr::Response &r, int attempt_number) {
    auto it = r.header.find("Retry-After");
    if (it != r.header.end()) {
      try {
        int retry_after_sec = std::stoi(it->second);
        if (retry_after_sec >= 0) {
          return std::min(retry_after_sec, 60) * 1000;
        }
      } catch (...) {
      }
    }
    return 500 * (1 << (attempt_number - 1));
  }
};

} // namespace llm_client
