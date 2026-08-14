#pragma once

#include <stdexcept>
#include <string>

namespace llm_client {

// 최상위 모듈 커스텀 예외
class LLMException : public std::runtime_error {
public:
  explicit LLMException(const std::string &message)
      : std::runtime_error(message) {}
};

// HTTP 네트워크 및 cpr 통신 에러
class NetworkException : public LLMException {
public:
  explicit NetworkException(const std::string &message)
      : LLMException("Network Error: " + message) {}
};

// nlohmann::json 파싱 및 구조 에러
class ParseException : public LLMException {
public:
  explicit ParseException(const std::string &message)
      : LLMException("Parse Error: " + message) {}
};

// API 응답 상의 상태 코드 에러 및 제공자 거부 에러
class APIException : public LLMException {
public:
  explicit APIException(const std::string &message)
      : LLMException("API Error: " + message) {}
};

// 설정이나 잘못된 파라미터 값으로 인한 오류
class ConfigurationException : public LLMException {
public:
  explicit ConfigurationException(const std::string &message)
      : LLMException("Configuration Error: " + message) {}
};

} // namespace llm_client
