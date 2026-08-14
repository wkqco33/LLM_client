#include "llm_client/llm_client_interface.hpp"
#include <iostream>
#include <cassert>
#include <chrono>
#include <thread>

// Mock LLM Client 구현체
class MockLLMClient : public llm_client::LLMClientInterface {
public:
    llm_client::ResponseData chat(const std::vector<llm_client::Message> &messages,
                                  const llm_client::RequestParams &params) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        llm_client::ResponseData res;
        res.content = "Mock response to: " + (messages.empty() ? "" : messages.back().content);
        res.total_tokens = 10;
        return res;
    }

    llm_client::ResponseData chatStream(const std::vector<llm_client::Message> &messages,
                                        llm_client::StreamCallback callback,
                                        const llm_client::RequestParams &params) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (callback) callback("Mock ");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (callback) callback("stream ");
        if (callback) callback("response");

        llm_client::ResponseData res;
        res.content = "Mock stream response";
        res.total_tokens = 15;
        return res;
    }
};

int main() {
    MockLLMClient client;

    std::cout << "[Test 1] generateAsync test..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    auto fut = client.generateAsync("Hello Async");

    // future가 준비될 때까지 비동기 작업 진행 확인
    assert(fut.valid());
    std::cout << "Async task launched without blocking." << std::endl;

    auto response = fut.get();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count();

    std::cout << "Response content: " << response.content << std::endl;
    std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;
    assert(response.content == "Mock response to: Hello Async");
    assert(elapsed >= 150);

    std::cout << "[Test 2] generateStreamAsync test..." << std::endl;
    std::string stream_out;
    auto fut_stream = client.generateStreamAsync("Hello Stream", [&stream_out](const std::string& chunk) {
        stream_out += chunk;
    });

    auto stream_res = fut_stream.get();
    std::cout << "Stream output: " << stream_out << std::endl;
    assert(stream_out == "Mock stream response");
    assert(stream_res.content == "Mock stream response");

    std::cout << "All async tests passed successfully!" << std::endl;
    return 0;
}
