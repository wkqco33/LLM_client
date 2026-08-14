#pragma once

#include "llm_client/exceptions.hpp"
#include "llm_client/logger.hpp"
#include "llm_client/types.hpp"
#include <algorithm>
#include <chrono>
#include <cpr/cpr.h>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace llm_client {
namespace detail {

// 429/5xx/네트워크 오류 등 재시도 대상 여부 판단 (스트리밍에서는 별도로
// "이미 콜백에 데이터가 전달됐는지" 조건이 추가로 붙는다)
inline bool is_retryable_response(const cpr::Response &r) {
  return r.error.code != cpr::ErrorCode::OK || r.status_code == 0 ||
         r.status_code == 429 || (r.status_code >= 500 && r.status_code < 600);
}

// 재시도 대기 시간 계산: 서버가 Retry-After 헤더(초 단위)를 주면 이를
// 우선하고, 없으면 지수 백오프(500ms, 1000ms, 2000ms, ...)를 사용한다.
// 서버가 비정상적으로 큰 값을 주는 경우를 대비해 최대 60초로 제한한다.
inline int compute_retry_delay_ms(const cpr::Response &r, int attempt_number) {
  auto it = r.header.find("Retry-After");
  if (it != r.header.end()) {
    try {
      int retry_after_sec = std::stoi(it->second);
      if (retry_after_sec >= 0) {
        return std::min(retry_after_sec, 60) * 1000;
      }
    } catch (const std::exception &) {
      // Retry-After 파싱 실패 시 지수 백오프로 폴백
    }
  }
  return 500 * (1 << (attempt_number - 1));
}

// 재시도 없이 최종 실패로 판단된 경우 적절한 예외를 던진다.
[[noreturn]] inline void throw_http_error(const cpr::Response &r,
                                          const std::string &provider_name,
                                          bool is_stream) {
  if (r.error.code != cpr::ErrorCode::OK || r.status_code == 0) {
    std::string err_msg = provider_name +
                          (is_stream ? " Stream Connection Error: "
                                     : " Network Connection Error: ") +
                          r.error.message;
    if (!is_stream) {
      std::string code_str = std::to_string(static_cast<int>(r.error.code));
      err_msg += " (code: " + code_str + ")";
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

// optional 값이 있을 때만 JSON 필드를 채운다. 대부분의 provider가
// temperature/top_p/max_tokens 등을 "값이 있으면 채우고 없으면 생략"하는
// 동일한 패턴을 반복하므로 공용화한다.
template <typename T>
inline void set_if_present(nlohmann::json &target, const char *key,
                           const std::optional<T> &opt) {
  if (opt.has_value()) {
    target[key] = *opt;
  }
}

// {"role": ..., "content": ...} 형태의 메시지 배열 생성. OpenAI/Azure OpenAI/
// Ollama처럼 role을 그대로 전달하는 provider들이 공유한다.
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

// SSE("data: {...}") 또는 접두사 없는 NDJSON 스트림 한 줄을 안전하게 파싱해
// on_json 콜백에 넘긴다.
// - data_prefix와 일치하지 않거나, 페이로드가 비어 있거나 "[DONE]"이면 조용히
//   건너뛴다 (로그 없음 — SSE의 빈 줄/이벤트 타입 줄 등 정상 케이스).
// - JSON 파싱은 물론 on_json 콜백(필드 추출) 내부에서 발생하는 예외까지 함께
//   포착해 경고 로그만 남기고 무시한다 (한 조각 파싱/스키마 오류로 전체 스트림이
//   중단되지 않도록). data_prefix를 빈 문자열로 넘기면 Ollama처럼 접두사 없는
//   라인 전체를 그대로 JSON으로 파싱한다.
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
    LLM_LOG_WARN("{}: failed to parse stream chunk (unknown error), "
                "skipping (raw: {})",
                provider_name, payload);
  }
}

/**
 * @brief 공통 HTTP POST 요청 및 지수 백오프 재시도/타임아웃 처리 헬퍼 함수
 */
inline nlohmann::json http_post_json(const std::string &url,
                                     const cpr::Header &headers,
                                     const nlohmann::json &req_json,
                                     const std::string &provider_name,
                                     int timeout_ms = 30000,
                                     int max_retries = 0) {
  auto start_time = std::chrono::high_resolution_clock::now();

  std::string req_body = req_json.dump();
  LLM_LOG_DEBUG("{} HTTP Request URL: {} (timeout: {}ms, max_retries: {})",
                provider_name, url, timeout_ms, max_retries);
  LLM_LOG_TRACE("{} HTTP Request Body: {}", provider_name, req_body);

  cpr::Response r;
  int attempts = 0;

  cpr::Session session;
  session.SetUrl(cpr::Url{url});
  session.SetHeader(headers);
  session.SetBody(cpr::Body{req_body});
  session.SetTimeout(cpr::Timeout{timeout_ms});

  while (true) {
    r = session.Post();

    if (r.error.code == cpr::ErrorCode::OK && r.status_code == 200) {
      break;
    }

    if (is_retryable_response(r) && attempts < max_retries) {
      attempts++;
      int delay_ms = compute_retry_delay_ms(r, attempts);
      LLM_LOG_WARN(
          "{} HTTP request failed (HTTP {}, error: {}). Retrying ({}/{}) in {}ms...",
          provider_name, r.status_code, r.error.message, attempts, max_retries,
          delay_ms);
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      continue;
    }

    throw_http_error(r, provider_name, /*is_stream=*/false);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

  LLM_LOG_DEBUG("{} API response received ({} ms), parsing JSON...",
                provider_name, elapsed_ms);

  // JSON 파싱 예외 안전 포획
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
 * @brief 공통 HTTP POST 스트리밍(SSE) 요청 및 재시도/타임아웃 처리 헬퍼 함수
 */
inline void http_post_stream(
    const std::string &url, const cpr::Header &headers,
    const nlohmann::json &req_json, const std::string &provider_name,
    std::function<void(const std::string &line)> on_line,
    int timeout_ms = 60000, int max_retries = 0) {
  auto start_time = std::chrono::high_resolution_clock::now();

  std::string req_body = req_json.dump();
  LLM_LOG_DEBUG("{} HTTP Stream Request URL: {} (timeout: {}ms, max_retries: {})",
                provider_name, url, timeout_ms, max_retries);
  LLM_LOG_TRACE("{} HTTP Stream Request Body: {}", provider_name, req_body);

  cpr::Response r;
  int attempts = 0;

  cpr::Session session;
  session.SetUrl(cpr::Url{url});
  session.SetHeader(headers);
  session.SetBody(cpr::Body{req_body});
  session.SetTimeout(cpr::Timeout{timeout_ms});

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
        on_line(line);
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
        any_line_received = true;
        on_line(line_buffer);
      }
      break;
    }

    // 이미 on_line(콜백)으로 일부 조각이 전달된 이후 연결이 끊긴 경우, 여기서
    // 재시도하면 서버가 스트림을 처음부터 다시 보내면서 콜백에 동일한 내용이
    // 중복 전달된다. 이미 호출자에게 전달된 데이터는 되돌릴 수 없으므로 이
    // 경우에는 재시도하지 않고 즉시 실패 처리한다.
    bool is_retryable = !any_line_received && is_retryable_response(r);

    if (any_line_received && r.error.code != cpr::ErrorCode::OK) {
      LLM_LOG_ERROR("{} Stream interrupted after partial data was already "
                    "delivered to the callback; aborting without retry to "
                    "avoid duplicate output (error: {})",
                    provider_name, r.error.message);
    }

    if (is_retryable && attempts < max_retries) {
      attempts++;
      int delay_ms = compute_retry_delay_ms(r, attempts);
      LLM_LOG_WARN(
          "{} HTTP stream request failed (HTTP {}, error: {}). Retrying ({}/{}) in {}ms...",
          provider_name, r.status_code, r.error.message, attempts, max_retries,
          delay_ms);
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      continue;
    }

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
