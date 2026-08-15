#pragma once

#include "llm_client/onnx_types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace llm_client {

/**
 * @brief ONNX 추론 엔진 추상 인터페이스
 */
class IOnnxEngine {
public:
  virtual ~IOnnxEngine() = default;

  /**
   * @brief ONNX 모델 로드
   * @param model_path 모델 파일 경로 (.onnx)
   * @return 로드 성공 여부
   */
  virtual bool loadModel(const std::string &model_path) = 0;

  /**
   * @brief 텐서 입력 추론 수행
   * @param inputs 입력 텐서 맵
   * @return 출력 텐서 맵
   */
  virtual OnnxTensorMap run(const OnnxTensorMap &inputs) = 0;

  /**
   * @brief 모델 로드 상태 확인
   */
  virtual bool isLoaded() const = 0;

  /**
   * @brief 모델 입력 텐서 이름 목록
   */
  virtual std::vector<std::string> getInputNames() const = 0;

  /**
   * @brief 모델 출력 텐서 이름 목록
   */
  virtual std::vector<std::string> getOutputNames() const = 0;

  /**
   * @brief 로드된 모델 경로
   */
  virtual std::string getModelPath() const = 0;
};

} // namespace llm_client
