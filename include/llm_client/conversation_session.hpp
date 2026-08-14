#pragma once

#include "llm_client/llm_client_interface.hpp"
#include "llm_client/types.hpp"
#include <string>
#include <vector>
#include <memory>

namespace llm_client {

/**
 * @brief 데이터 중심의 대화 세션 컨텍스트 관리 클래스
 */
class ConversationSession {
public:
    explicit ConversationSession(std::shared_ptr<LLMClientInterface> client,
                                std::string system_prompt = "",
                                size_t max_history_messages = 20)
        : client_(std::move(client)), max_history_(max_history_messages) {
        if (!system_prompt.empty()) {
            messages_.push_back(Message{"system", std::move(system_prompt)});
        }
    }

    void addMessage(Message msg) {
        messages_.push_back(std::move(msg));
        trimHistoryIfNeeded();
    }

    void addUserMessage(std::string text) {
        addMessage(Message{"user", std::move(text)});
    }

    ResponseData send(const RequestParams &params = {}) {
        ResponseData resp = client_->chat(messages_, params);
        if (!resp.content.empty()) {
            messages_.push_back(Message{"assistant", resp.content});
            trimHistoryIfNeeded();
        }
        return resp;
    }

    ResponseData sendStream(StreamCallback callback, const RequestParams &params = {}) {
        ResponseData resp = client_->chatStream(messages_, std::move(callback), params);
        if (!resp.content.empty()) {
            messages_.push_back(Message{"assistant", resp.content});
            trimHistoryIfNeeded();
        }
        return resp;
    }

    const std::vector<Message>& getMessages() const { return messages_; }
    
    void clearHistory(bool keep_system_prompt = true) {
        if (keep_system_prompt && !messages_.empty() && messages_.front().role == "system") {
            Message sys = messages_.front();
            messages_.clear();
            messages_.push_back(std::move(sys));
        } else {
            messages_.clear();
        }
    }

private:
    void trimHistoryIfNeeded() {
        if (max_history_ == 0) return;
        
        bool has_system = (!messages_.empty() && messages_.front().role == "system");
        size_t effective_max = max_history_ + (has_system ? 1 : 0);
        
        while (messages_.size() > effective_max) {
            auto it = has_system ? messages_.begin() + 1 : messages_.begin();
            messages_.erase(it);
        }
    }

    std::shared_ptr<LLMClientInterface> client_;
    std::vector<Message> messages_;
    size_t max_history_;
};

} // namespace llm_client
