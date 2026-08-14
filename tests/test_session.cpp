#include "llm_client/conversation_session.hpp"
#include <iostream>
#include <cassert>

class MockLLMClient : public llm_client::LLMClientInterface {
public:
    llm_client::ResponseData chat(const std::vector<llm_client::Message> &messages,
                                  const llm_client::RequestParams &params) override {
        llm_client::ResponseData res;
        res.content = "Reply to " + (messages.empty() ? "" : messages.back().content);
        return res;
    }

    llm_client::ResponseData chatStream(const std::vector<llm_client::Message> &messages,
                                        llm_client::StreamCallback callback,
                                        const llm_client::RequestParams &params) override {
        if (callback) callback("Stream reply");
        llm_client::ResponseData res;
        res.content = "Stream reply";
        return res;
    }
};

int main() {
    std::cout << "[Test ConversationSession]" << std::endl;
    auto client = std::make_shared<MockLLMClient>();

    // max_history = 3 (system 메시지 제외 메시지 수 3개)
    llm_client::ConversationSession session(client, "You are a helpful assistant.", 3);

    assert(session.getMessages().size() == 1);
    assert(session.getMessages()[0].role == "system");

    session.addUserMessage("Msg 1");
    auto res1 = session.send();
    assert(res1.content == "Reply to Msg 1");

    session.addUserMessage("Msg 2");
    auto res2 = session.send();

    session.addUserMessage("Msg 3");
    auto res3 = session.send();

    // 메시지 트리밍 확인 (system 메시지는 보존되고 슬라이딩 윈도우 적용됨)
    const auto& msgs = session.getMessages();
    std::cout << "Message count after trim: " << msgs.size() << std::endl;
    assert(msgs.front().role == "system");
    assert(msgs.size() <= 4);

    session.clearHistory(/*keep_system_prompt=*/true);
    assert(session.getMessages().size() == 1);
    assert(session.getMessages()[0].role == "system");

    std::cout << "ConversationSession test passed successfully!" << std::endl;
    return 0;
}
