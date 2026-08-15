#include "llm_client/exceptions.hpp"
#include "llm_client/llm_client_interface.hpp"
#include <gtest/gtest.h>

using namespace llm_client;

class MockEmbeddingClient : public LLMClientInterface {
public:
  ResponseData chat(const std::vector<Message> &,
                    const RequestParams &) override {
    return {};
  }

  ResponseData chatStream(const std::vector<Message> &,
                          StreamCallback,
                          const RequestParams &) override {
    return {};
  }

  EmbeddingResponse embed(const std::vector<std::string> &inputs,
                          const EmbeddingParams &params = {}) override {
    EmbeddingResponse res;
    res.model = params.model.empty() ? "mock-embedding" : params.model;
    for (size_t i = 0; i < inputs.size(); ++i) {
      res.embeddings.push_back({static_cast<float>(i), static_cast<float>(inputs[i].size())});
    }
    res.prompt_tokens = static_cast<int>(inputs.size());
    res.total_tokens = res.prompt_tokens;
    return res;
  }
};

class MockUnsupportedClient : public LLMClientInterface {
public:
  ResponseData chat(const std::vector<Message> &,
                    const RequestParams &) override {
    return {};
  }

  ResponseData chatStream(const std::vector<Message> &,
                          StreamCallback,
                          const RequestParams &) override {
    return {};
  }
};

TEST(EmbeddingTest, EmbedBasic) {
  MockEmbeddingClient client;
  std::vector<std::string> inputs = {"hello", "world!"};

  auto res = client.embed(inputs);
  EXPECT_EQ(res.embeddings.size(), inputs.size());
  ASSERT_FALSE(res.embeddings.empty());
  EXPECT_EQ(res.embeddings[0].size(), 2);
  EXPECT_EQ(res.model, "mock-embedding");
  EXPECT_EQ(res.total_tokens, 2);
}

TEST(EmbeddingTest, EmbedAsync) {
  MockEmbeddingClient client;
  std::vector<std::string> inputs = {"hello", "world!"};

  auto fut = client.embedAsync(inputs);
  ASSERT_TRUE(fut.valid());
  auto async_res = fut.get();
  EXPECT_EQ(async_res.embeddings.size(), inputs.size());
}

TEST(EmbeddingTest, DefaultEmbedThrowsConfigurationException) {
  MockUnsupportedClient unsupported_client;
  std::vector<std::string> inputs = {"hello"};

  EXPECT_THROW(unsupported_client.embed(inputs), ConfigurationException);
}
