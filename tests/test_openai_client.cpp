#include "llm_client/openai_client.hpp"
#include "llm_client/exceptions.hpp"
#include "llm_client/mock_http_client.hpp"
#include <gtest/gtest.h>

using namespace llm_client;

class OpenAIClientTest : public ::testing::Test {
protected:
  std::shared_ptr<MockHttpClient> mock_http;

  void SetUp() override {
    mock_http = std::make_shared<MockHttpClient>();
  }
};

TEST_F(OpenAIClientTest, ChatSuccess_ParsesResponseAndUsage) {
  mock_http->setResponse(200, R"({
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o-2024-05-13",
    "choices": [{
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
      },
      "finish_reason": "stop"
    }],
    "usage": {
      "prompt_tokens": 9,
      "completion_tokens": 12,
      "total_tokens": 21
    }
  })");

  OpenAIClient client("test-key", "https://api.openai.com/v1", mock_http);
  ResponseData res = client.generate("Hello");

  EXPECT_EQ(res.content, "Hello! How can I assist you today?");
  EXPECT_EQ(res.finish_reason, "stop");
  EXPECT_EQ(res.model, "gpt-4o-2024-05-13");
  EXPECT_EQ(res.prompt_tokens, 9);
  EXPECT_EQ(res.completion_tokens, 12);
  EXPECT_EQ(res.total_tokens, 21);

  // 요청 URL, Header, Body 검증
  EXPECT_EQ(mock_http->getLastUrl(), "https://api.openai.com/v1/chat/completions");
  auto headers = mock_http->getLastHeaders();
  EXPECT_EQ(headers["Authorization"], "Bearer test-key");
  EXPECT_EQ(headers["Content-Type"], "application/json");

  auto json_body = mock_http->getLastJsonBody();
  EXPECT_EQ(json_body["model"], "gpt-4o");
  EXPECT_EQ(json_body["messages"].size(), 1);
  EXPECT_EQ(json_body["messages"][0]["role"], "user");
  EXPECT_EQ(json_body["messages"][0]["content"], "Hello");
}

TEST_F(OpenAIClientTest, ChatStream_CollectsChunksAndFinishReason) {
  std::vector<std::string> sse_lines = {
      "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
      "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}",
      "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
      "data: [DONE]"
  };
  mock_http->setStreamLines(sse_lines, 200);

  OpenAIClient client("test-key", "https://api.openai.com/v1", mock_http);

  std::string accumulated;
  ResponseData res = client.generateStream("Stream test", [&](const std::string &chunk) {
    accumulated += chunk;
  });

  EXPECT_EQ(accumulated, "Hello world");
  EXPECT_EQ(res.content, "Hello world");
  EXPECT_EQ(res.finish_reason, "stop");
  EXPECT_TRUE(mock_http->getLastRequest().is_stream);
}

TEST_F(OpenAIClientTest, Chat_Handles429RateLimitThrowsAPIException) {
  mock_http->setResponse(429, R"({"error": {"message": "Rate limit reached"}})");

  OpenAIClient client("test-key", "https://api.openai.com/v1", mock_http);
  EXPECT_THROW(client.generate("Hello"), APIException);
}

TEST_F(OpenAIClientTest, Chat_HandlesNetworkErrorThrowsNetworkException) {
  mock_http->setError(7, "Failed to connect to host");

  OpenAIClient client("test-key", "https://api.openai.com/v1", mock_http);
  EXPECT_THROW(client.generate("Hello"), NetworkException);
}

TEST_F(OpenAIClientTest, Chat_HandlesInvalidJsonThrowsParseException) {
  mock_http->setResponse(200, "This is not a JSON text {");

  OpenAIClient client("test-key", "https://api.openai.com/v1", mock_http);
  EXPECT_THROW(client.generate("Hello"), ParseException);
}

TEST_F(OpenAIClientTest, EmbedSuccess_ParsesEmbeddingsAndUsage) {
  mock_http->setResponse(200, R"({
    "object": "list",
    "data": [
      {
        "object": "embedding",
        "index": 0,
        "embedding": [0.1, 0.2, 0.3]
      },
      {
        "object": "embedding",
        "index": 1,
        "embedding": [0.4, 0.5, 0.6]
      }
    ],
    "model": "text-embedding-3-small",
    "usage": {
      "prompt_tokens": 8,
      "total_tokens": 8
    }
  })");

  OpenAIClient client("test-key", "https://api.openai.com/v1", mock_http);
  EmbeddingResponse res = client.embed({"text 1", "text 2"});

  EXPECT_EQ(res.embeddings.size(), 2);
  EXPECT_EQ(res.embeddings[0].size(), 3);
  EXPECT_FLOAT_EQ(res.embeddings[0][0], 0.1f);
  EXPECT_FLOAT_EQ(res.embeddings[1][2], 0.6f);
  EXPECT_EQ(res.total_tokens, 8);
  EXPECT_EQ(mock_http->getLastUrl(), "https://api.openai.com/v1/embeddings");
}

TEST_F(OpenAIClientTest, RequestParams_ResponseFormatJsonSchema) {
  mock_http->setResponse(200, R"({
    "choices": [{"message": {"content": "{\"result\": 42}"}, "finish_reason": "stop"}],
    "model": "gpt-4o"
  })");

  OpenAIClient client("test-key", "https://api.openai.com/v1", mock_http);
  RequestParams params;
  params.response_format = "json_schema";
  params.json_schema = R"({"type": "object", "properties": {"result": {"type": "integer"}}})";
  params.temperature = 0.7f;
  params.max_tokens = 100;

  client.generate("Schema test", params);

  auto body = mock_http->getLastJsonBody();
  EXPECT_TRUE(body.contains("response_format"));
  EXPECT_EQ(body["response_format"]["type"], "json_schema");
  EXPECT_FLOAT_EQ(body["temperature"].get<float>(), 0.7f);
  EXPECT_EQ(body["max_tokens"], 100);
}
