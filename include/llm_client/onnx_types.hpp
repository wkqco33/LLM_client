#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace llm_client {

/**
 * @brief ONNX 텐서 차원 크기
 */
using TensorShape = std::vector<int64_t>;

/**
 * @brief 텐서 데이터 컨테이너 (float 및 int64 데이터 지원)
 */
struct TensorData {
  enum class DataType { Float, Int64 };

  DataType type = DataType::Float;
  TensorShape shape;
  std::vector<float> float_data;
  std::vector<int64_t> int64_data;

  static TensorData createFloat(TensorShape shape, std::vector<float> data) {
    TensorData t;
    t.type = DataType::Float;
    t.shape = std::move(shape);
    t.float_data = std::move(data);
    return t;
  }

  static TensorData createInt64(TensorShape shape, std::vector<int64_t> data) {
    TensorData t;
    t.type = DataType::Int64;
    t.shape = std::move(shape);
    t.int64_data = std::move(data);
    return t;
  }

  size_t totalElements() const {
    if (shape.empty()) return 0;
    size_t count = 1;
    for (auto dim : shape) {
      count *= static_cast<size_t>(dim > 0 ? dim : 1);
    }
    return count;
  }
};

/**
 * @brief ONNX 입출력 텐서 맵
 */
using OnnxTensorMap = std::map<std::string, TensorData>;

} // namespace llm_client
