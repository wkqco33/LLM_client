#include "llm_client/azure_openai_client.hpp"
#include "llm_client/exceptions.hpp"
#include "llm_client/llm_client_factory.hpp"
#include "llm_client/logger.hpp"
#include "llm_client/ollama_client.hpp"
#include "llm_client/openai_client.hpp"

namespace llm_client {

std::unique_ptr<LLMClientInterface>
LLMClientFactory::create(const std::string &provider,
                         const std::string &api_key,
                         const std::string &base_url,
                         const std::string &api_version) {
  if (provider == "openai") {
    return std::make_unique<OpenAIClient>(
        api_key, base_url.empty() ? "https://api.openai.com/v1" : base_url);
  } else if (provider == "azure") {
    if (base_url.empty()) {
      LLM_LOG_ERROR("Base URL is empty for azure provider");
      throw ConfigurationException("Azure provider requires a base_url.");
    }
    // Azure OpenAI는 리소스마다 고유한 Endpoint 형식이 다릅니다.
    // 일단 사용자가 base_url로 Endpoint, API버전은 환경변수나 하드코딩 등
    // 규칙에 따르지만 여기서는 기본적으로 api_version을 고정하거나, Factory
    // API를 확장해야 할 수 있습니다. 기존 create 함수 구조를 유지하기 위해,
    // api_version을 "2024-02-15-preview" 처럼 기본값으로 넣습니다.
    return std::make_unique<AzureOpenAIClient>(
        api_key, api_version.empty() ? "2024-02-15-preview" : api_version,
        base_url);
  } else if (provider == "ollama") {
    return std::make_unique<OllamaClient>(
        base_url.empty() ? "http://localhost:11434" : base_url, api_key);
  } else if (provider == "custom") {
    if (base_url.empty()) {
      LLM_LOG_ERROR("Base URL is empty for custom provider");
      throw ConfigurationException("Custom provider requires a base_url.");
    }
    return std::make_unique<OpenAIClient>(api_key, base_url);
  } else {
    LLM_LOG_ERROR("Unknown LLM provider requested: {}", provider);
    throw ConfigurationException("Unknown LLM provider: " + provider);
  }
}

} // namespace llm_client
