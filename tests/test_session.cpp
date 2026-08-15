#include "llm_client/conversation_session.hpp"
#include "llm_client/llm_client_interface.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace llm_client;

class MockSessionLLMClient : public LLMClientInterface {
public:
  ResponseData chat(const std::vector<Message> &messages,
                    const RequestParams &) override {
    ResponseData res;
    res.content = "Reply to " + (messages.empty() ? "" : messages.back().content);
    return res;
  }

  ResponseData chatStream(const std::vector<Message> &messages,
                          StreamCallback callback,
                          const RequestParams &) override {
    if (callback) {
      callback("Stream reply");
    }
    ResponseData res;
    res.content = "Stream reply";
    return res;
  }
};

TEST(ConversationSessionTest, InitialSystemPrompt) {
  auto client = std::make_shared<MockSessionLLMClient>();
  ConversationSession session(client, "You are a helpful assistant.", 5);

  const auto &msgs = session.getMessages();
  ASSERT_EQ(msgs.size(), 1);
  EXPECT_EQ(msgs[0].role, "system");
  EXPECT_EQ(msgs[0].content, "You are a helpful assistant.");
}

TEST(ConversationSessionTest, SendAndAutoAppendAssistantResponse) {
  auto client = std::make_shared<MockSessionLLMClient>();
  ConversationSession session(client, "System", 5);

  session.addUserMessage("Msg 1");
  auto res1 = session.send();
  EXPECT_EQ(res1.content, "Reply to Msg 1");

  const auto &msgs = session.getMessages();
  ASSERT_EQ(msgs.size(), 3);
  EXPECT_EQ(msgs[1].role, "user");
  EXPECT_EQ(msgs[1].content, "Msg 1");
  EXPECT_EQ(msgs[2].role, "assistant");
  EXPECT_EQ(msgs[2].content, "Reply to Msg 1");
}

TEST(ConversationSessionTest, SendStreamAndAutoAppend) {
  auto client = std::make_shared<MockSessionLLMClient>();
  ConversationSession session(client, "System", 5);

  session.addUserMessage("Stream message");
  std::string stream_out;
  auto res = session.sendStream([&](const std::string &chunk) {
    stream_out += chunk;
  });

  EXPECT_EQ(stream_out, "Stream reply");
  EXPECT_EQ(res.content, "Stream reply");

  const auto &msgs = session.getMessages();
  ASSERT_EQ(msgs.size(), 3);
  EXPECT_EQ(msgs.back().role, "assistant");
  EXPECT_EQ(msgs.back().content, "Stream reply");
}

TEST(ConversationSessionTest, SlidingWindowTrimmingPreservesSystemPrompt) {
  auto client = std::make_shared<MockSessionLLMClient>();
  // max_history = 3 (system 메시지 제외 메시지 최대 3개)
  ConversationSession session(client, "System prompt", 3);

  session.addUserMessage("Msg 1");
  session.send(); // msgs: system, user1, assist1 (3)

  session.addUserMessage("Msg 2");
  session.send(); // msgs: system, user1, assist1, user2, assist2 -> trim -> system, assist1, user2, assist2 (4)

  const auto &msgs = session.getMessages();
  EXPECT_EQ(msgs.front().role, "system");
  EXPECT_EQ(msgs.front().content, "System prompt");
  EXPECT_LE(msgs.size(), 4);
}

TEST(ConversationSessionTest, ClearHistory) {
  auto client = std::make_shared<MockSessionLLMClient>();
  ConversationSession session(client, "System prompt", 5);

  session.addUserMessage("Msg 1");
  session.send();

  session.clearHistory(/*keep_system_prompt=*/true);
  ASSERT_EQ(session.getMessages().size(), 1);
  EXPECT_EQ(session.getMessages()[0].role, "system");

  session.clearHistory(/*keep_system_prompt=*/false);
  EXPECT_TRUE(session.getMessages().empty());
}
