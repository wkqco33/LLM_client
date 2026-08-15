#pragma once

#include "llm_client/http_client_interface.hpp"
#include "llm_client/llm_client_interface.hpp"
#include <memory>
#include <string>

namespace llm_client {

class LLMClientFactory {
public:
  // 프로바이더 이름(예: "openai", "ollama")과 API 키 지정
  // azure 제공자는 api_version을 추가 지정할 수 있습니다.
  // http_client를 지정하지 않으면 기본 CprHttpClient를 사용합니다.
  static std::unique_ptr<LLMClientInterface>
  create(const std::string &provider, const std::string &api_key,
         const std::string &base_url = "",
         const std::string &api_version = "",
         std::shared_ptr<IHttpClient> http_client = nullptr);
};

} // namespace llm_client
