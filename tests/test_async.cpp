#include "llm_client/llm_client_interface.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <thread>

using namespace llm_client;

class MockAsyncLLMClient : public LLMClientInterface {
public:
  ResponseData chat(const std::vector<Message> &messages,
                    const RequestParams &) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ResponseData res;
    res.content = "Mock response to: " + (messages.empty() ? "" : messages.back().content);
    res.total_tokens = 10;
    return res;
  }

  ResponseData chatStream(const std::vector<Message> &,
                          StreamCallback callback,
                          const RequestParams &) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    if (callback) callback("Mock ");
    if (callback) callback("stream ");
    if (callback) callback("response");

    ResponseData res;
    res.content = "Mock stream response";
    res.total_tokens = 15;
    return res;
  }
};

TEST(AsyncTest, GenerateAsync) {
  MockAsyncLLMClient client;
  auto fut = client.generateAsync("Hello Async");

  ASSERT_TRUE(fut.valid());
  auto response = fut.get();

  EXPECT_EQ(response.content, "Mock response to: Hello Async");
  EXPECT_EQ(response.total_tokens, 10);
}

TEST(AsyncTest, GenerateStreamAsync) {
  MockAsyncLLMClient client;
  std::string stream_out;
  auto fut_stream = client.generateStreamAsync("Hello Stream", [&](const std::string &chunk) {
    stream_out += chunk;
  });

  ASSERT_TRUE(fut_stream.valid());
  auto stream_res = fut_stream.get();

  EXPECT_EQ(stream_out, "Mock stream response");
  EXPECT_EQ(stream_res.content, "Mock stream response");
  EXPECT_EQ(stream_res.total_tokens, 15);
}
