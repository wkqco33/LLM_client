#pragma once

#include "llm_client/http_client_interface.hpp"
#include <deque>
#include <functional>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm_client {

/**
 * @brief TDD 단위 테스트용 가짜(Mock) HTTP 클라이언트
 */
class MockHttpClient : public IHttpClient {
public:
  struct RecordedRequest {
    std::string url;
    std::map<std::string, std::string> headers;
    std::string body;
    int timeout_ms = 0;
    int max_retries = 0;
    bool is_stream = false;
  };

  using PostHandler = std::function<HttpResponse(const std::string &url,
                                                 const std::map<std::string, std::string> &headers,
                                                 const std::string &body)>;
  using StreamHandler = std::function<HttpResponse(const std::string &url,
                                                   const std::map<std::string, std::string> &headers,
                                                   const std::string &body,
                                                   std::function<void(const std::string &line)> on_line)>;

  MockHttpClient() = default;
  ~MockHttpClient() override = default;

  // 단일 기본 응답 설정
  void setResponse(int status_code, const std::string &text,
                   const std::map<std::string, std::string> &headers = {}) {
    default_response_.status_code = status_code;
    default_response_.text = text;
    default_response_.headers = headers;
    default_response_.error_code = 0;
    default_response_.error_message = "";
  }

  // 네트워크 에러 시뮬레이션
  void setError(int error_code, const std::string &error_message) {
    default_response_.status_code = 0;
    default_response_.text = "";
    default_response_.error_code = error_code;
    default_response_.error_message = error_message;
  }

  // FIFO 응답 큐에 추가
  void queueResponse(const HttpResponse &resp) {
    response_queue_.push_back(resp);
  }

  // 스트리밍 라인 목록 설정 (SSE 청크 등)
  void setStreamLines(const std::vector<std::string> &lines, int status_code = 200) {
    default_stream_lines_ = lines;
    default_response_.status_code = status_code;
  }

  void queueStreamLines(const std::vector<std::string> &lines, int status_code = 200) {
    stream_lines_queue_.push_back({status_code, lines});
  }

  void setPostHandler(PostHandler handler) { post_handler_ = std::move(handler); }
  void setStreamHandler(StreamHandler handler) { stream_handler_ = std::move(handler); }

  HttpResponse post(const std::string &url,
                    const std::map<std::string, std::string> &headers,
                    const std::string &body, int timeout_ms = 30000,
                    int max_retries = 0) override {
    recorded_requests_.push_back({url, headers, body, timeout_ms, max_retries, /*is_stream=*/false});

    if (post_handler_) {
      return post_handler_(url, headers, body);
    }

    if (!response_queue_.empty()) {
      HttpResponse resp = response_queue_.front();
      response_queue_.pop_front();
      return resp;
    }

    return default_response_;
  }

  HttpResponse postStream(const std::string &url,
                          const std::map<std::string, std::string> &headers,
                          const std::string &body,
                          std::function<void(const std::string &line)> on_line,
                          int timeout_ms = 60000,
                          int max_retries = 0) override {
    recorded_requests_.push_back({url, headers, body, timeout_ms, max_retries, /*is_stream=*/true});

    if (stream_handler_) {
      return stream_handler_(url, headers, body, on_line);
    }

    std::vector<std::string> lines_to_emit = default_stream_lines_;
    int status_code = default_response_.status_code;

    if (!stream_lines_queue_.empty()) {
      auto item = stream_lines_queue_.front();
      stream_lines_queue_.pop_front();
      status_code = item.first;
      lines_to_emit = item.second;
    }

    if (on_line) {
      for (const auto &line : lines_to_emit) {
        on_line(line);
      }
    }

    HttpResponse resp = default_response_;
    resp.status_code = status_code;
    return resp;
  }

  // 검증용 헬퍼 함수들
  const std::vector<RecordedRequest> &getRecordedRequests() const {
    return recorded_requests_;
  }

  size_t getRequestCount() const { return recorded_requests_.size(); }

  const RecordedRequest &getLastRequest() const {
    if (recorded_requests_.empty()) {
      throw std::runtime_error("No requests have been recorded in MockHttpClient");
    }
    return recorded_requests_.back();
  }

  std::string getLastUrl() const { return getLastRequest().url; }
  std::string getLastBody() const { return getLastRequest().body; }

  nlohmann::json getLastJsonBody() const {
    return nlohmann::json::parse(getLastBody());
  }

  std::map<std::string, std::string> getLastHeaders() const {
    return getLastRequest().headers;
  }

  void clearHistory() {
    recorded_requests_.clear();
    response_queue_.clear();
    stream_lines_queue_.clear();
  }

private:
  HttpResponse default_response_{200, "{}", {}, 0, ""};
  std::vector<std::string> default_stream_lines_;
  std::deque<HttpResponse> response_queue_;
  std::deque<std::pair<int, std::vector<std::string>>> stream_lines_queue_;
  std::vector<RecordedRequest> recorded_requests_;
  PostHandler post_handler_;
  StreamHandler stream_handler_;
};

} // namespace llm_client
