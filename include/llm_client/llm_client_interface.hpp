#pragma once

#include "llm_client/exceptions.hpp"
#include "llm_client/types.hpp"
#include <string>
#include <vector>

#include <future>
#include <utility>

namespace llm_client {

class LLMClientInterface {
public:
  virtual ~LLMClientInterface() = default;

  // 대화 이력(메시지 목록)을 받아 응답을 생성하는 메인 인터페이스
  virtual ResponseData chat(const std::vector<Message> &messages,
                            const RequestParams &params = {}) = 0;

  // 스트리밍 방식으로 실시간 응답 조각을 콜백으로 수신하는 인터페이스
  virtual ResponseData chatStream(const std::vector<Message> &messages,
                                  StreamCallback callback,
                                  const RequestParams &params = {}) = 0;

  // 비동기 (Future 기반) chat
  virtual std::future<ResponseData> chatAsync(const std::vector<Message> &messages,
                                               const RequestParams &params = {}) {
    return std::async(std::launch::async, [this, messages, params]() {
      return chat(messages, params);
    });
  }

  // 비동기 (Future 기반) chatStream
  virtual std::future<ResponseData> chatStreamAsync(const std::vector<Message> &messages,
                                                     StreamCallback callback,
                                                     const RequestParams &params = {}) {
    return std::async(std::launch::async, [this, messages, callback = std::move(callback), params]() {
      return chatStream(messages, callback, params);
    });
  }

  // 단일 프롬프트 스트링을 바로 입력해서 응답을 생성하는 편의 함수
  virtual ResponseData generate(const std::string &prompt,
                                const RequestParams &params = {}) {
    return chat({{"user", prompt}}, params);
  }

  // 단일 프롬프트를 스트리밍 방식으로 처리하는 편의 함수
  virtual ResponseData generateStream(const std::string &prompt,
                                      StreamCallback callback,
                                      const RequestParams &params = {}) {
    return chatStream({{"user", prompt}}, callback, params);
  }

  // 비동기 generate 편의 함수
  virtual std::future<ResponseData> generateAsync(const std::string &prompt,
                                                  const RequestParams &params = {}) {
    return chatAsync({{"user", prompt}}, params);
  }

  // 비동기 generateStream 편의 함수
  virtual std::future<ResponseData> generateStreamAsync(const std::string &prompt,
                                                        StreamCallback callback,
                                                        const RequestParams &params = {}) {
    return chatStreamAsync({{"user", prompt}}, std::move(callback), params);
  }

  // API 연결 상태 혹은 인증 정보 유효성 검사 (옵션)
  virtual bool validateConnection() { return true; }

  // 텍스트 입력을 임베딩 벡터로 변환 (RAG 등에서 사용). 기본적으로는
  // 미지원 예외를 던지며, 임베딩 API를 제공하는 프로바이더(OpenAI, Azure
  // OpenAI, Ollama 등)만 오버라이드해서 구현한다.
  virtual EmbeddingResponse embed(const std::vector<std::string> &inputs,
                                  const EmbeddingParams &params = {}) {
    (void)inputs;
    (void)params;
    throw ConfigurationException("embed() is not supported by this provider");
  }

  // 비동기 (Future 기반) embed
  virtual std::future<EmbeddingResponse> embedAsync(const std::vector<std::string> &inputs,
                                                     const EmbeddingParams &params = {}) {
    return std::async(std::launch::async, [this, inputs, params]() {
      return embed(inputs, params);
    });
  }
};

} // namespace llm_client
