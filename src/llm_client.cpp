#include "llm_client/llm_client_factory.hpp"
#include "llm_client/azure_openai_client.hpp"
#include "llm_client/exceptions.hpp"
#include "llm_client/logger.hpp"
#include "llm_client/ollama_client.hpp"
#include "llm_client/onnx_client.hpp"
#include "llm_client/openai_client.hpp"

namespace llm_client {

std::unique_ptr<LLMClientInterface>
LLMClientFactory::create(const std::string &provider,
                         const std::string &api_key,
                         const std::string &base_url,
                         const std::string &api_version,
                         std::shared_ptr<IHttpClient> http_client) {
  if (provider == "openai") {
    return std::make_unique<OpenAIClient>(
        api_key, base_url.empty() ? "https://api.openai.com/v1" : base_url,
        std::move(http_client));
  } else if (provider == "azure") {
    if (base_url.empty()) {
      LLM_LOG_ERROR("Base URL is empty for azure provider");
      throw ConfigurationException("Azure provider requires a base_url.");
    }
    return std::make_unique<AzureOpenAIClient>(
        api_key, api_version.empty() ? "2024-02-15-preview" : api_version,
        base_url, std::move(http_client));
  } else if (provider == "ollama") {
    return std::make_unique<OllamaClient>(
        base_url.empty() ? "http://localhost:11434" : base_url, api_key,
        std::move(http_client));
  } else if (provider == "onnx") {
    if (base_url.empty()) {
      LLM_LOG_ERROR("Model path (base_url) is empty for onnx provider");
      throw ConfigurationException("ONNX provider requires a model path in base_url.");
    }
    // base_url = model_path, api_version = tokenizer_path (옵션)
    return std::make_unique<OnnxClient>(base_url, api_version);
  } else if (provider == "custom") {
    if (base_url.empty()) {
      LLM_LOG_ERROR("Base URL is empty for custom provider");
      throw ConfigurationException("Custom provider requires a base_url.");
    }
    return std::make_unique<OpenAIClient>(api_key, base_url, std::move(http_client));
  } else {
    LLM_LOG_ERROR("Unknown LLM provider requested: {}", provider);
    throw ConfigurationException("Unknown LLM provider: " + provider);
  }
}

} // namespace llm_client
