#include "llm_client/simple_tokenizer.hpp"
#include "llm_client/logger.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace llm_client {

SimpleTokenizer::SimpleTokenizer() {
  addToken("<pad>", pad_token_id_);
  addToken("<s>", bos_token_id_);
  addToken("</s>", eos_token_id_);
  addToken("<unk>", unk_token_id_);
}

SimpleTokenizer::SimpleTokenizer(const std::map<std::string, int64_t> &vocab) {
  token_to_id_ = vocab;
  for (const auto &[k, v] : vocab) {
    id_to_token_[v] = k;
  }
}

void SimpleTokenizer::addToken(const std::string &token, int64_t id) {
  token_to_id_[token] = id;
  id_to_token_[id] = token;
}

bool SimpleTokenizer::loadVocabFromJson(const std::string &json_str) {
  try {
    auto j = nlohmann::json::parse(json_str);
    token_to_id_.clear();
    id_to_token_.clear();

    if (j.contains("model") && j["model"].contains("vocab")) {
      // HuggingFace tokenizer.json 포맷
      for (auto &[token, id_val] : j["model"]["vocab"].items()) {
        int64_t id = id_val.get<int64_t>();
        addToken(token, id);
      }
    } else if (j.is_object()) {
      // 일반 단어-ID 맵 포맷
      for (auto &[token, id_val] : j.items()) {
        if (id_val.is_number()) {
          int64_t id = id_val.get<int64_t>();
          addToken(token, id);
        }
      }
    }
    LLM_LOG_INFO("SimpleTokenizer: loaded {} tokens from JSON", token_to_id_.size());
    return true;
  } catch (const std::exception &e) {
    LLM_LOG_WARN("SimpleTokenizer: failed to parse vocab json: {}", e.what());
    return false;
  }
}

bool SimpleTokenizer::loadVocabFromFile(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    LLM_LOG_WARN("SimpleTokenizer: cannot open file {}", file_path);
    return false;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return loadVocabFromJson(buffer.str());
}

TokenizedPrompt SimpleTokenizer::encode(const std::string &text) const {
  TokenizedPrompt prompt;
  std::istringstream iss(text);
  std::string word;

  while (iss >> word) {
    auto it = token_to_id_.find(word);
    if (it != token_to_id_.end()) {
      prompt.input_ids.push_back(it->second);
    } else {
      prompt.input_ids.push_back(unk_token_id_);
    }
    prompt.attention_mask.push_back(1);
    prompt.token_type_ids.push_back(0);
  }

  if (prompt.input_ids.empty()) {
    prompt.input_ids.push_back(unk_token_id_);
    prompt.attention_mask.push_back(1);
    prompt.token_type_ids.push_back(0);
  }

  return prompt;
}

std::string SimpleTokenizer::decode(const std::vector<int64_t> &tokens) const {
  std::string result;
  for (size_t i = 0; i < tokens.size(); ++i) {
    int64_t tok = tokens[i];
    if (tok == eos_token_id_ || tok == pad_token_id_ || tok == bos_token_id_) {
      continue;
    }
    std::string word = decodeToken(tok);
    if (!result.empty() && !word.empty()) {
      result += " ";
    }
    result += word;
  }
  return result;
}

std::string SimpleTokenizer::decodeToken(int64_t token_id) const {
  auto it = id_to_token_.find(token_id);
  if (it != id_to_token_.end()) {
    return it->second;
  }
  return "<unk>";
}

} // namespace llm_client
