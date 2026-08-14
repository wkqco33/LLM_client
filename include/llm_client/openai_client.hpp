#pragma once

#include "llm_client/llm_client_interface.hpp"
#include <string>
#include <vector>

namespace llm_client {

class OpenAIClient : public LLMClientInterface {
public:
  OpenAIClient(const std::string &api_key,
               const std::string &base_url = "https://api.openai.com/v1");
  ~OpenAIClient() override = default;

  ResponseData chat(const std::vector<Message> &messages,
                    const RequestParams &params = {}) override;
  ResponseData chatStream(const std::vector<Message> &messages,
                          StreamCallback callback,
                          const RequestParams &params = {}) override;
  EmbeddingResponse embed(const std::vector<std::string> &inputs,
                         const EmbeddingParams &params = {}) override;

private:
  std::string api_key_;
  std::string base_url_;
};

} // namespace llm_client
