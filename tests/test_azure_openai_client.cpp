#include "llm_client/azure_openai_client.hpp"
#include "llm_client/exceptions.hpp"
#include "llm_client/mock_http_client.hpp"
#include <gtest/gtest.h>

using namespace llm_client;

class AzureOpenAIClientTest : public ::testing::Test {
protected:
  std::shared_ptr<MockHttpClient> mock_http;

  void SetUp() override {
    mock_http = std::make_shared<MockHttpClient>();
  }
};

TEST_F(AzureOpenAIClientTest, Chat_UsesDeploymentNameInUrlAndApiKeyHeader) {
  mock_http->setResponse(200, R"({
    "id": "chatcmpl-azure-123",
    "choices": [{
      "message": {"role": "assistant", "content": "Hello from Azure!"},
      "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
  })");

  AzureOpenAIClient client("azure-secret-key", "2024-02-15-preview",
                           "https://my-resource.openai.azure.com", mock_http);

  RequestParams params;
  params.model = "my-gpt4-deployment";
  ResponseData res = client.generate("Hello Azure", params);

  EXPECT_EQ(res.content, "Hello from Azure!");
  EXPECT_EQ(res.total_tokens, 11);

  // URL 및 헤더 검증
  EXPECT_EQ(mock_http->getLastUrl(),
            "https://my-resource.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions?api-version=2024-02-15-preview");
  auto headers = mock_http->getLastHeaders();
  EXPECT_EQ(headers["api-key"], "azure-secret-key");
  EXPECT_EQ(headers["Content-Type"], "application/json");
}

TEST_F(AzureOpenAIClientTest, Chat_ReasoningModelUsesMaxCompletionTokens) {
  mock_http->setResponse(200, R"({
    "choices": [{"message": {"role": "assistant", "content": "Reasoned response"}, "finish_reason": "stop"}]
  })");

  AzureOpenAIClient client("azure-secret-key", "2024-02-15-preview",
                           "https://my-resource.openai.azure.com", mock_http);

  RequestParams params;
  params.model = "o1-mini-deployment"; // o1 접두사 -> max_completion_tokens 사용
  params.max_tokens = 500;

  client.generate("Think deeply", params);

  auto body = mock_http->getLastJsonBody();
  EXPECT_TRUE(body.contains("max_completion_tokens"));
  EXPECT_FALSE(body.contains("max_tokens"));
  EXPECT_EQ(body["max_completion_tokens"], 500);
}

TEST_F(AzureOpenAIClientTest, Embed_UsesAzureDeploymentUrl) {
  mock_http->setResponse(200, R"({
    "data": [{"embedding": [0.1, 0.2], "index": 0}],
    "usage": {"prompt_tokens": 4, "total_tokens": 4}
  })");

  AzureOpenAIClient client("azure-secret-key", "2024-02-15-preview",
                           "https://my-resource.openai.azure.com", mock_http);

  EmbeddingParams params;
  params.model = "text-embedding-ada";
  EmbeddingResponse res = client.embed({"Test input"}, params);

  EXPECT_EQ(res.embeddings.size(), 1);
  EXPECT_EQ(mock_http->getLastUrl(),
            "https://my-resource.openai.azure.com/openai/deployments/text-embedding-ada/embeddings?api-version=2024-02-15-preview");
}
