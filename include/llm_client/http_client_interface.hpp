#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llm_client {

/**
 * @brief HTTP 요청에 대한 응답 구조체
 */
struct HttpResponse {
  int status_code = 0;
  std::string text;
  std::map<std::string, std::string> headers;
  int error_code = 0; // 0이면 정상, 0이 아니면 네트워크 에러 (예: cpr::ErrorCode)
  std::string error_message;
};

/**
 * @brief HTTP 통신 추상화 인터페이스
 *
 * 단위 테스트 시 가짜 응답을 주입(Mocking)하거나,
 * 다른 통신 백엔드로 교체할 수 있도록 합니다.
 */
class IHttpClient {
public:
  virtual ~IHttpClient() = default;

  /**
   * @brief 동기 HTTP POST 요청
   * @param url 요청 대상 URL
   * @param headers HTTP 헤더 목록
   * @param body 요청 본문 (JSON 등)
   * @param timeout_ms 타임아웃 (밀리초)
   * @param max_retries 재시도 횟수
   * @return HttpResponse 응답 구조체
   */
  virtual HttpResponse
  post(const std::string &url,
       const std::map<std::string, std::string> &headers,
       const std::string &body, int timeout_ms = 30000,
       int max_retries = 0) = 0;

  /**
   * @brief 스트리밍 HTTP POST 요청 (SSE / NDJSON 라인 콜백)
   * @param url 요청 대상 URL
   * @param headers HTTP 헤더 목록
   * @param body 요청 본문
   * @param on_line 수신된 라인별 콜백 함수
   * @param timeout_ms 타임아웃 (밀리초)
   * @param max_retries 재시도 횟수
   * @return HttpResponse 최종 응답 정보
   */
  virtual HttpResponse
  postStream(const std::string &url,
             const std::map<std::string, std::string> &headers,
             const std::string &body,
             std::function<void(const std::string &line)> on_line,
             int timeout_ms = 60000, int max_retries = 0) = 0;
};

} // namespace llm_client
