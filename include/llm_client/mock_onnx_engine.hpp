#pragma once

#include "llm_client/onnx_engine_interface.hpp"
#include <deque>
#include <functional>
#include <stdexcept>

namespace llm_client {

/**
 * @brief TDD 단위 테스트용 Mock ONNX 추론 엔진
 */
class MockOnnxEngine : public IOnnxEngine {
public:
  using Handler = std::function<OnnxTensorMap(const OnnxTensorMap &inputs)>;

  MockOnnxEngine() : is_loaded_(true), input_names_{"input_ids", "attention_mask"}, output_names_{"logits"} {}
  ~MockOnnxEngine() override = default;

  bool loadModel(const std::string &model_path) override {
    model_path_ = model_path;
    is_loaded_ = !model_path.empty();
    return is_loaded_;
  }

  OnnxTensorMap run(const OnnxTensorMap &inputs) override {
    if (!is_loaded_) {
      throw std::runtime_error("MockOnnxEngine: model is not loaded");
    }
    recorded_inputs_.push_back(inputs);

    if (handler_) {
      return handler_(inputs);
    }

    if (!output_queue_.empty()) {
      auto out = output_queue_.front();
      output_queue_.pop_front();
      return out;
    }

    return default_output_;
  }

  bool isLoaded() const override { return is_loaded_; }
  std::vector<std::string> getInputNames() const override { return input_names_; }
  std::vector<std::string> getOutputNames() const override { return output_names_; }
  std::string getModelPath() const override { return model_path_; }

  // Mock 설정 메서드들
  void setLoaded(bool loaded) { is_loaded_ = loaded; }
  void setDefaultOutput(OnnxTensorMap output) { default_output_ = std::move(output); }
  void queueOutput(OnnxTensorMap output) { output_queue_.push_back(std::move(output)); }
  void setHandler(Handler handler) { handler_ = std::move(handler); }
  void setInputNames(std::vector<std::string> names) { input_names_ = std::move(names); }
  void setOutputNames(std::vector<std::string> names) { output_names_ = std::move(names); }

  // 검증 메서드들
  const std::vector<OnnxTensorMap> &getRecordedInputs() const { return recorded_inputs_; }
  size_t getRunCount() const { return recorded_inputs_.size(); }
  const OnnxTensorMap &getLastInputs() const {
    if (recorded_inputs_.empty()) {
      throw std::runtime_error("No inferences recorded in MockOnnxEngine");
    }
    return recorded_inputs_.back();
  }
  void clearHistory() {
    recorded_inputs_.clear();
    output_queue_.clear();
  }

private:
  bool is_loaded_ = false;
  std::string model_path_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  OnnxTensorMap default_output_;
  std::deque<OnnxTensorMap> output_queue_;
  Handler handler_;
  std::vector<OnnxTensorMap> recorded_inputs_;
};

} // namespace llm_client
