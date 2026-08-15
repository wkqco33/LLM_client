#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm_client {

/**
 * @brief 토크나이징 인코딩 결과 구조체
 */
struct TokenizedPrompt {
  std::vector<int64_t> input_ids;
  std::vector<int64_t> attention_mask;
  std::vector<int64_t> token_type_ids; // BERT 계열 등 옵션
};

/**
 * @brief 토크나이저 추상 인터페이스
 */
class ITokenizer {
public:
  virtual ~ITokenizer() = default;

  virtual TokenizedPrompt encode(const std::string &text) const = 0;
  virtual std::string decode(const std::vector<int64_t> &tokens) const = 0;
  virtual std::string decodeToken(int64_t token_id) const = 0;

  virtual int64_t getEosTokenId() const = 0;
  virtual int64_t getBosTokenId() const = 0;
  virtual int64_t getPadTokenId() const = 0;
  virtual size_t getVocabSize() const = 0;
};

} // namespace llm_client
