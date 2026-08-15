#pragma once

#include "llm_client/tokenizer_interface.hpp"
#include <map>
#include <string>
#include <vector>

namespace llm_client {

/**
 * @brief 사전(Vocab) 및 공백/규칙 기반 기본 토크나이저
 */
class SimpleTokenizer : public ITokenizer {
public:
  SimpleTokenizer();
  explicit SimpleTokenizer(const std::map<std::string, int64_t> &vocab);

  bool loadVocabFromJson(const std::string &json_str);
  bool loadVocabFromFile(const std::string &file_path);

  TokenizedPrompt encode(const std::string &text) const override;
  std::string decode(const std::vector<int64_t> &tokens) const override;
  std::string decodeToken(int64_t token_id) const override;

  int64_t getEosTokenId() const override { return eos_token_id_; }
  int64_t getBosTokenId() const override { return bos_token_id_; }
  int64_t getPadTokenId() const override { return pad_token_id_; }
  size_t getVocabSize() const override { return token_to_id_.size(); }

  void setSpecialTokens(int64_t bos, int64_t eos, int64_t pad, int64_t unk) {
    bos_token_id_ = bos;
    eos_token_id_ = eos;
    pad_token_id_ = pad;
    unk_token_id_ = unk;
  }

  void addToken(const std::string &token, int64_t id);

private:
  std::map<std::string, int64_t> token_to_id_;
  std::map<int64_t, std::string> id_to_token_;
  int64_t bos_token_id_ = 1;
  int64_t eos_token_id_ = 2;
  int64_t pad_token_id_ = 0;
  int64_t unk_token_id_ = 3;
};

} // namespace llm_client
