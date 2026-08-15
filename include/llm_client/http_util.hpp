#pragma once

#include "llm_client/exceptions.hpp"
#include "llm_client/http_client_interface.hpp"
#include "llm_client/logger.hpp"
#include "llm_client/types.hpp"
#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace llm_client {
namespace detail {

// 재시도 없이 최종 실패로 판단된 경우 적절한 예외를 던진다.
[[noreturn]] inline void throw_http_error(const HttpResponse &r,
                                          const std::string &provider_name,
                                          bool is_stream) {
  if (r.error_code != 0 || r.status_code == 0) {
    std::string err_msg = provider_name +
                          (is_stream ? " Stream Connection Error: "
                                     : " Network Connection Error: ") +
                          r.error_message;
    if (!is_stream && r.error_code != 0) {
      err_msg += " (code: " + std::to_string(r.error_code) + ")";
    }
    LLM_LOG_ERROR(err_msg);
    throw NetworkException(err_msg);
  }

  std::string err_msg = provider_name +
                        (is_stream ? " Stream API fail: HTTP " : " API fail: HTTP ") +
                        std::to_string(r.status_code) + " - " + r.text;
  LLM_LOG_ERROR(err_msg);
  throw APIException(err_msg);
}

// optional 값이 있을 때만 JSON 필드를 채운다.
template <typename T>
inline void set_if_present(nlohmann::json &target, const char *key,
                           const std::optional<T> &opt) {
  if (opt.has_value()) {
    target[key] = *opt;
  }
}

// {"role": ..., "content": ...} 형태의 메시지 배열 생성
inline nlohmann::json
build_role_content_messages(const std::vector<Message> &messages) {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto &msg : messages) {
    if (msg.blocks.empty()) {
      arr.push_back({{"role", msg.role}, {"content", msg.content}});
    } else {
      nlohmann::json content_arr = nlohmann::json::array();
      for (const auto &block : msg.blocks) {
        if (block.type == ContentType::Text) {
          content_arr.push_back({{"type", "text"}, {"text", block.text}});
        } else if (block.type == ContentType::ImageUrl) {
          content_arr.push_back({{"type", "image_url"}, {"image_url", {{"url", block.image_url}}}});
        } else if (block.type == ContentType::ImageBase64) {
          std::string data_url = "data:" + block.media_type + ";base64," + block.base64_data;
          content_arr.push_back({{"type", "image_url"}, {"image_url", {{"url", data_url}}}});
        }
      }
      arr.push_back({{"role", msg.role}, {"content", content_arr}});
    }
  }
  return arr;
}

// SSE("data: {...}") 또는 NDJSON 스트림 한 줄 파싱
inline void
parse_stream_json_line(const std::string &line, const std::string &provider_name,
                       const std::function<void(const nlohmann::json &)> &on_json,
                       const std::string &data_prefix = "data: ") {
  std::string payload = line;
  if (!data_prefix.empty()) {
    if (line.rfind(data_prefix, 0) != 0) {
      return;
    }
    payload = line.substr(data_prefix.size());
  }
  if (payload.empty() || payload == "[DONE]") {
    return;
  }

  try {
    nlohmann::json j = nlohmann::json::parse(payload);
    on_json(j);
  } catch (const std::exception &e) {
    LLM_LOG_WARN("{}: failed to parse stream chunk, skipping: {} (raw: {})",
                provider_name, e.what(), payload);
  } catch (...) {
    LLM_LOG_WARN("{}: failed to parse stream chunk (unknown error), skipping (raw: {})",
                provider_name, payload);
  }
}

/**
 * @brief IHttpClient를 사용하는 공통 HTTP POST 요청 및 예외/JSON 파싱 처리
 */
inline nlohmann::json http_post_json(IHttpClient &http_client,
                                     const std::string &url,
                                     const std::map<std::string, std::string> &headers,
                                     const nlohmann::json &req_json,
                                     const std::string &provider_name,
                                     int timeout_ms = 30000,
                                     int max_retries = 0) {
  auto start_time = std::chrono::high_resolution_clock::now();

  std::string req_body = req_json.dump();
  LLM_LOG_DEBUG("{} HTTP Request URL: {} (timeout: {}ms, max_retries: {})",
                provider_name, url, timeout_ms, max_retries);
  LLM_LOG_TRACE("{} HTTP Request Body: {}", provider_name, req_body);

  HttpResponse r = http_client.post(url, headers, req_body, timeout_ms, max_retries);

  if (r.error_code != 0 || r.status_code != 200) {
    throw_http_error(r, provider_name, /*is_stream=*/false);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

  LLM_LOG_DEBUG("{} API response received ({} ms), parsing JSON...",
                provider_name, elapsed_ms);

  try {
    return nlohmann::json::parse(r.text);
  } catch (const nlohmann::json::parse_error &e) {
    std::string err_msg = provider_name +
                          " API response JSON parsing error: " + e.what() +
                          " (Response text: " + r.text + ")";
    LLM_LOG_ERROR(err_msg);
    throw ParseException(err_msg);
  }
}

/**
 * @brief IHttpClient를 사용하는 공통 HTTP POST 스트리밍(SSE) 요청 및 예외 처리
 */
inline void http_post_stream(
    IHttpClient &http_client,
    const std::string &url,
    const std::map<std::string, std::string> &headers,
    const nlohmann::json &req_json,
    const std::string &provider_name,
    std::function<void(const std::string &line)> on_line,
    int timeout_ms = 60000,
    int max_retries = 0) {
  auto start_time = std::chrono::high_resolution_clock::now();

  std::string req_body = req_json.dump();
  LLM_LOG_DEBUG("{} HTTP Stream Request URL: {} (timeout: {}ms, max_retries: {})",
                provider_name, url, timeout_ms, max_retries);
  LLM_LOG_TRACE("{} HTTP Stream Request Body: {}", provider_name, req_body);

  HttpResponse r = http_client.postStream(url, headers, req_body, on_line, timeout_ms, max_retries);

  if (r.error_code != 0 || r.status_code != 200) {
    throw_http_error(r, provider_name, /*is_stream=*/true);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

  LLM_LOG_DEBUG("{} Stream completed ({} ms)", provider_name, elapsed_ms);
}

} // namespace detail
} // namespace llm_client
