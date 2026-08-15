#include "llm_client/ollama_client.hpp"
#include "llm_client/exceptions.hpp"
#include "llm_client/mock_http_client.hpp"
#include <gtest/gtest.h>

using namespace llm_client;

class OllamaClientTest : public ::testing::Test {
protected:
  std::shared_ptr<MockHttpClient> mock_http;

  void SetUp() override {
    mock_http = std::make_shared<MockHttpClient>();
  }
};

TEST_F(OllamaClientTest, ChatSuccess_ParsesThinkingAndDoneReason) {
  mock_http->setResponse(200, R"({
    "model": "deepseek-r1:8b",
    "message": {
      "role": "assistant",
      "content": "42",
      "thinking": "Thinking about the answer..."
    },
    "done_reason": "stop",
    "prompt_eval_count": 15,
    "eval_count": 25
  })");

  OllamaClient client("http://localhost:11434", "", mock_http);
  RequestParams params;
  params.model = "deepseek-r1:8b";
  ResponseData res = client.generate("What is 6 * 7?", params);

  EXPECT_EQ(res.content, "42");
  EXPECT_EQ(res.reasoning_content, "Thinking about the answer...");
  EXPECT_EQ(res.finish_reason, "stop");
  EXPECT_EQ(res.prompt_tokens, 15);
  EXPECT_EQ(res.completion_tokens, 25);
  EXPECT_EQ(res.total_tokens, 40);
  EXPECT_EQ(mock_http->getLastUrl(), "http://localhost:11434/api/chat");
}

TEST_F(OllamaClientTest, ChatStream_StreamsContentAndReasoning) {
  std::vector<std::string> ndjson_lines = {
      "{\"message\":{\"role\":\"assistant\",\"content\":\"Hello\",\"thinking\":\"Thinking 1\"},\"done\":false}",
      "{\"message\":{\"role\":\"assistant\",\"content\":\" world\",\"thinking\":\"Thinking 2\"},\"done\":false}",
      "{\"message\":{\"role\":\"assistant\",\"content\":\"!\"},\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":10,\"eval_count\":20}"
  };
  mock_http->setStreamLines(ndjson_lines, 200);

  OllamaClient client("http://localhost:11434", "", mock_http);

  std::string accumulated;
  ResponseData res = client.generateStream("Stream test", [&](const std::string &chunk) {
    accumulated += chunk;
  });

  EXPECT_EQ(accumulated, "Hello world!");
  EXPECT_EQ(res.content, "Hello world!");
  EXPECT_EQ(res.reasoning_content, "Thinking 1Thinking 2");
  EXPECT_EQ(res.finish_reason, "stop");
  EXPECT_EQ(res.total_tokens, 30);
}

TEST_F(OllamaClientTest, OptionsThinkingNumPredict) {
  mock_http->setResponse(200, R"({
    "model": "llama3",
    "message": {"content": "OK"}
  })");

  OllamaClient client("http://localhost:11434", "bearer-token", mock_http);

  RequestParams params;
  params.temperature = 0.5f;
  params.max_tokens = 128;
  params.thinking = true;

  client.generate("Test options", params);

  auto body = mock_http->getLastJsonBody();
  EXPECT_TRUE(body.contains("options"));
  EXPECT_FLOAT_EQ(body["options"]["temperature"].get<float>(), 0.5f);
  EXPECT_EQ(body["options"]["num_predict"], 128);
  EXPECT_TRUE(body["options"]["think"]);
  EXPECT_TRUE(body["think"]);

  auto headers = mock_http->getLastHeaders();
  EXPECT_EQ(headers["Authorization"], "Bearer bearer-token");
}

TEST_F(OllamaClientTest, EmbedSuccess_ParsesEmbeddings) {
  mock_http->setResponse(200, R"({
    "model": "nomic-embed-text",
    "embeddings": [
      [0.01, 0.02, 0.03],
      [0.04, 0.05, 0.06]
    ],
    "prompt_eval_count": 12
  })");

  OllamaClient client("http://localhost:11434", "", mock_http);
  EmbeddingResponse res = client.embed({"first", "second"});

  EXPECT_EQ(res.embeddings.size(), 2);
  EXPECT_FLOAT_EQ(res.embeddings[0][0], 0.01f);
  EXPECT_FLOAT_EQ(res.embeddings[1][2], 0.06f);
  EXPECT_EQ(res.total_tokens, 12);
  EXPECT_EQ(mock_http->getLastUrl(), "http://localhost:11434/api/embed");
}
