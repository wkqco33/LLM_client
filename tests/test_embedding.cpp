#include "llm_client/llm_client_interface.hpp"
#include <cassert>
#include <iostream>

// 임베딩을 지원하는 Mock LLM Client 구현체
class MockEmbeddingClient : public llm_client::LLMClientInterface {
public:
    llm_client::ResponseData chat(const std::vector<llm_client::Message> &,
                                  const llm_client::RequestParams &) override {
        return {};
    }

    llm_client::ResponseData chatStream(const std::vector<llm_client::Message> &,
                                        llm_client::StreamCallback,
                                        const llm_client::RequestParams &) override {
        return {};
    }

    llm_client::EmbeddingResponse embed(const std::vector<std::string> &inputs,
                                        const llm_client::EmbeddingParams &params = {}) override {
        llm_client::EmbeddingResponse res;
        res.model = params.model.empty() ? "mock-embedding" : params.model;
        for (size_t i = 0; i < inputs.size(); ++i) {
            res.embeddings.push_back({static_cast<float>(i), static_cast<float>(inputs[i].size())});
        }
        res.prompt_tokens = static_cast<int>(inputs.size());
        res.total_tokens = res.prompt_tokens;
        return res;
    }
};

// 임베딩 미지원 프로바이더를 흉내내는 Mock (인터페이스 기본 동작 검증용)
class MockUnsupportedClient : public llm_client::LLMClientInterface {
public:
    llm_client::ResponseData chat(const std::vector<llm_client::Message> &,
                                  const llm_client::RequestParams &) override {
        return {};
    }

    llm_client::ResponseData chatStream(const std::vector<llm_client::Message> &,
                                        llm_client::StreamCallback,
                                        const llm_client::RequestParams &) override {
        return {};
    }
};

int main() {
    std::cout << "[Test 1] embed() basic test..." << std::endl;
    MockEmbeddingClient client;
    std::vector<std::string> inputs = {"hello", "world!"};

    auto res = client.embed(inputs);
    assert(res.embeddings.size() == inputs.size());
    assert(res.embeddings[0].size() == 2);
    assert(res.model == "mock-embedding");
    assert(res.total_tokens == 2);
    std::cout << "embed() returned " << res.embeddings.size() << " embeddings." << std::endl;

    std::cout << "[Test 2] embedAsync() test..." << std::endl;
    auto fut = client.embedAsync(inputs);
    assert(fut.valid());
    auto async_res = fut.get();
    assert(async_res.embeddings.size() == inputs.size());
    std::cout << "embedAsync() completed with " << async_res.embeddings.size()
               << " embeddings." << std::endl;

    std::cout << "[Test 3] default embed() throws ConfigurationException..." << std::endl;
    MockUnsupportedClient unsupported_client;
    bool threw = false;
    try {
        unsupported_client.embed(inputs);
    } catch (const llm_client::ConfigurationException &) {
        threw = true;
    }
    assert(threw);
    std::cout << "Unsupported provider correctly threw ConfigurationException." << std::endl;

    std::cout << "All embedding tests passed successfully!" << std::endl;
    return 0;
}
