#pragma once

#include "llm_client/llm_client_interface.hpp"
#include <string>

namespace llm_client {

class OllamaClient : public LLMClientInterface {
public:
  explicit OllamaClient(const std::string &base_url = "http://localhost:11434",
                        const std::string &api_key = "");

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
};

} // namespace llm_client
