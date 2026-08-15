#pragma once

#include "llm_client/http_client_interface.hpp"
#include "llm_client/llm_client_interface.hpp"
#include <memory>
#include <string>
#include <vector>

namespace llm_client {

class AzureOpenAIClient : public LLMClientInterface {
public:
  // Azure OpenAI는 api_key, api_version, base_url 모두 필수인 경우가 많습니다.
  AzureOpenAIClient(const std::string &api_key, const std::string &api_version,
                    const std::string &base_url,
                    std::shared_ptr<IHttpClient> http_client = nullptr);
  ~AzureOpenAIClient() override = default;

  ResponseData chat(const std::vector<Message> &messages,
                    const RequestParams &params = {}) override;
  ResponseData chatStream(const std::vector<Message> &messages,
                          StreamCallback callback,
                          const RequestParams &params = {}) override;
  EmbeddingResponse embed(const std::vector<std::string> &inputs,
                         const EmbeddingParams &params = {}) override;

private:
  std::string api_key_;
  std::string api_version_;
  std::string base_url_;
  std::shared_ptr<IHttpClient> http_client_;
};

} // namespace llm_client
