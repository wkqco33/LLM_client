#pragma once

#include "llm_client/http_client_interface.hpp"
#include "llm_client/llm_client_interface.hpp"
#include <memory>
#include <string>

namespace llm_client {

class OllamaClient : public LLMClientInterface {
public:
  explicit OllamaClient(const std::string &base_url = "http://localhost:11434",
                        const std::string &api_key = "",
                        std::shared_ptr<IHttpClient> http_client = nullptr);

  ResponseData chat(const std::vector<Message> &messages,
                    const RequestParams &params = {}) override;
  ResponseData chatStream(const std::vector<Message> &messages,
                          StreamCallback callback,
                          const RequestParams &params = {}) override;
  EmbeddingResponse embed(const std::vector<std::string> &inputs,
                         const EmbeddingParams &params = {}) override;

private:
  std::string base_url_;
  std::string api_key_;
  std::shared_ptr<IHttpClient> http_client_;
};

} // namespace llm_client
