#pragma once

#include "llm_client/llm_client_interface.hpp"
#include "llm_client/onnx_engine_interface.hpp"
#include "llm_client/tokenizer_interface.hpp"
#include <memory>
#include <string>
#include <vector>

namespace llm_client {

/**
 * @brief 로컬 ONNX 모델 추론 클라이언트
 */
class OnnxClient : public LLMClientInterface {
public:
  /**
   * @brief OnnxClient 생성자
   * @param model_path .onnx 모델 파일 경로
   * @param tokenizer_path 토크나이저 설정 파일 경로 (tokenizer.json, vocab.json 등)
   * @param engine ONNX 엔진 구현체 (미지정 시 기본 엔진 사용)
   * @param tokenizer 토크나이저 구현체 (미지정 시 SimpleTokenizer 사용)
   */
  explicit OnnxClient(const std::string &model_path,
                      const std::string &tokenizer_path = "",
                      std::shared_ptr<IOnnxEngine> engine = nullptr,
                      std::shared_ptr<ITokenizer> tokenizer = nullptr);

  ~OnnxClient() override = default;

  ResponseData chat(const std::vector<Message> &messages,
                    const RequestParams &params = {}) override;

  ResponseData chatStream(const std::vector<Message> &messages,
                          StreamCallback callback,
                          const RequestParams &params = {}) override;

  EmbeddingResponse embed(const std::vector<std::string> &inputs,
                          const EmbeddingParams &params = {}) override;

  bool validateConnection() override;

  // 접근자
  const std::string &getModelPath() const { return model_path_; }
  std::shared_ptr<IOnnxEngine> getEngine() const { return engine_; }
  std::shared_ptr<ITokenizer> getTokenizer() const { return tokenizer_; }

private:
  std::string model_path_;
  std::string tokenizer_path_;
  std::shared_ptr<IOnnxEngine> engine_;
  std::shared_ptr<ITokenizer> tokenizer_;

  // 내부 헬퍼 함수들
  std::string buildPromptFromMessages(const std::vector<Message> &messages) const;
  int64_t sampleNextToken(const std::vector<float> &logits, float temperature, float top_p) const;
  std::vector<float> extractLastTokenLogits(const TensorData &logits_tensor) const;
  std::vector<float> computeMeanPooling(const TensorData &hidden_state, const std::vector<int64_t> &attention_mask) const;
};

} // namespace llm_client
