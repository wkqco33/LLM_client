#include "llm_client/onnx_client.hpp"
#include "llm_client/exceptions.hpp"
#include "llm_client/logger.hpp"
#include "llm_client/mock_onnx_engine.hpp"
#include "llm_client/simple_tokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace llm_client {

OnnxClient::OnnxClient(const std::string &model_path,
                       const std::string &tokenizer_path,
                       std::shared_ptr<IOnnxEngine> engine,
                       std::shared_ptr<ITokenizer> tokenizer)
    : model_path_(model_path), tokenizer_path_(tokenizer_path),
      engine_(std::move(engine)), tokenizer_(std::move(tokenizer)) {
  if (!tokenizer_) {
    auto st = std::make_shared<SimpleTokenizer>();
    if (!tokenizer_path_.empty()) {
      st->loadVocabFromFile(tokenizer_path_);
    }
    tokenizer_ = st;
  }

  if (!engine_) {
    // 기본 Mock 엔진 (외부 바이너리/런타임 바인딩용 플레이스홀더)
    auto mock = std::make_shared<MockOnnxEngine>();
    if (!model_path_.empty()) {
      mock->loadModel(model_path_);
    }
    engine_ = mock;
  }
}

bool OnnxClient::validateConnection() {
  return engine_ && engine_->isLoaded();
}

std::string OnnxClient::buildPromptFromMessages(const std::vector<Message> &messages) const {
  std::string prompt;
  for (const auto &msg : messages) {
    if (!prompt.empty()) {
      prompt += " ";
    }
    if (!msg.content.empty()) {
      prompt += msg.content;
    } else {
      for (const auto &b : msg.blocks) {
        if (b.type == ContentType::Text) {
          prompt += b.text + " ";
        }
      }
    }
  }
  return prompt;
}

std::vector<float> OnnxClient::extractLastTokenLogits(const TensorData &logits_tensor) const {
  if (logits_tensor.float_data.empty() || logits_tensor.shape.empty()) {
    throw APIException("OnnxClient: logits tensor is empty");
  }

  // logits shape는 보통 [batch, seq_len, vocab_size]
  int64_t vocab_size = logits_tensor.shape.back();
  if (vocab_size <= 0) {
    vocab_size = static_cast<int64_t>(logits_tensor.float_data.size());
  }

  size_t total = logits_tensor.float_data.size();
  size_t start_idx = total >= static_cast<size_t>(vocab_size) ? total - static_cast<size_t>(vocab_size) : 0;

  return std::vector<float>(logits_tensor.float_data.begin() + start_idx,
                            logits_tensor.float_data.end());
}

int64_t OnnxClient::sampleNextToken(const std::vector<float> &logits, float temperature, float top_p) const {
  if (logits.empty()) {
    return tokenizer_->getEosTokenId();
  }

  // Greedy Search (temperature <= 0 또는 기본)
  if (temperature <= 0.01f) {
    auto max_it = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_it);
  }

  // Softmax with temperature
  std::vector<float> scaled_logits = logits;
  float max_val = *std::max_element(scaled_logits.begin(), scaled_logits.end());
  float sum_exp = 0.0f;
  for (auto &v : scaled_logits) {
    v = std::exp((v - max_val) / temperature);
    sum_exp += v;
  }
  for (auto &v : scaled_logits) {
    v /= sum_exp;
  }

  // Top-P (Nucleus Sampling)
  if (top_p > 0.0f && top_p < 1.0f) {
    std::vector<size_t> indices(scaled_logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
      return scaled_logits[a] > scaled_logits[b];
    });

    float cumsum = 0.0f;
    size_t cutoff = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      cumsum += scaled_logits[indices[i]];
      cutoff = i;
      if (cumsum >= top_p) break;
    }

    std::vector<float> filtered_probs(cutoff + 1);
    float norm_sum = 0.0f;
    for (size_t i = 0; i <= cutoff; ++i) {
      filtered_probs[i] = scaled_logits[indices[i]];
      norm_sum += filtered_probs[i];
    }
    for (auto &p : filtered_probs) p /= norm_sum;

    static std::mt19937 gen(1337);
    std::discrete_distribution<size_t> dist(filtered_probs.begin(), filtered_probs.end());
    return static_cast<int64_t>(indices[dist(gen)]);
  }

  static std::mt19937 gen(1337);
  std::discrete_distribution<size_t> dist(scaled_logits.begin(), scaled_logits.end());
  return static_cast<int64_t>(dist(gen));
}

ResponseData OnnxClient::chat(const std::vector<Message> &messages,
                              const RequestParams &params) {
  if (!engine_ || !engine_->isLoaded()) {
    LLM_LOG_ERROR("OnnxClient: ONNX model is not loaded: {}", model_path_);
    throw ConfigurationException("OnnxClient: model is not loaded: " + model_path_);
  }

  std::string prompt_text = buildPromptFromMessages(messages);
  TokenizedPrompt encoded = tokenizer_->encode(prompt_text);

  std::vector<int64_t> current_input_ids = encoded.input_ids;
  std::vector<int64_t> current_mask = encoded.attention_mask;
  std::vector<int64_t> generated_tokens;

  int max_new_tokens = params.max_tokens.value_or(128);
  float temperature = params.temperature.value_or(0.0f);
  float top_p = params.top_p.value_or(1.0f);

  LLM_LOG_INFO("OnnxClient: starting text generation for prompt (tokens: {})", current_input_ids.size());

  for (int step = 0; step < max_new_tokens; ++step) {
    TensorShape shape = {1, static_cast<int64_t>(current_input_ids.size())};
    OnnxTensorMap inputs;
    inputs["input_ids"] = TensorData::createInt64(shape, current_input_ids);
    inputs["attention_mask"] = TensorData::createInt64(shape, current_mask);

    OnnxTensorMap outputs;
    try {
      outputs = engine_->run(inputs);
    } catch (const std::exception &e) {
      LLM_LOG_ERROR("OnnxClient: engine execution error: {}", e.what());
      throw APIException(std::string("OnnxClient execution error: ") + e.what());
    }

    if (outputs.find("logits") == outputs.end()) {
      LLM_LOG_ERROR("OnnxClient: output tensor map does not contain 'logits'");
      throw ParseException("OnnxClient: 'logits' tensor not found in model output");
    }

    std::vector<float> last_logits = extractLastTokenLogits(outputs.at("logits"));
    int64_t next_token = sampleNextToken(last_logits, temperature, top_p);

    generated_tokens.push_back(next_token);

    if (next_token == tokenizer_->getEosTokenId()) {
      break;
    }

    current_input_ids.push_back(next_token);
    current_mask.push_back(1);
  }

  ResponseData resp;
  resp.content = tokenizer_->decode(generated_tokens);
  resp.model = model_path_;
  resp.finish_reason = "stop";
  resp.prompt_tokens = static_cast<int>(encoded.input_ids.size());
  resp.completion_tokens = static_cast<int>(generated_tokens.size());
  resp.total_tokens = resp.prompt_tokens + resp.completion_tokens;

  LLM_LOG_INFO("OnnxClient: generation finished (generated tokens: {})", generated_tokens.size());
  return resp;
}

ResponseData OnnxClient::chatStream(const std::vector<Message> &messages,
                                    StreamCallback callback,
                                    const RequestParams &params) {
  if (!engine_ || !engine_->isLoaded()) {
    throw ConfigurationException("OnnxClient: model is not loaded: " + model_path_);
  }

  std::string prompt_text = buildPromptFromMessages(messages);
  TokenizedPrompt encoded = tokenizer_->encode(prompt_text);

  std::vector<int64_t> current_input_ids = encoded.input_ids;
  std::vector<int64_t> current_mask = encoded.attention_mask;
  std::vector<int64_t> generated_tokens;

  int max_new_tokens = params.max_tokens.value_or(128);
  float temperature = params.temperature.value_or(0.0f);
  float top_p = params.top_p.value_or(1.0f);

  for (int step = 0; step < max_new_tokens; ++step) {
    TensorShape shape = {1, static_cast<int64_t>(current_input_ids.size())};
    OnnxTensorMap inputs;
    inputs["input_ids"] = TensorData::createInt64(shape, current_input_ids);
    inputs["attention_mask"] = TensorData::createInt64(shape, current_mask);

    OnnxTensorMap outputs = engine_->run(inputs);
    if (outputs.find("logits") == outputs.end()) {
      throw ParseException("OnnxClient: 'logits' tensor not found");
    }

    std::vector<float> last_logits = extractLastTokenLogits(outputs.at("logits"));
    int64_t next_token = sampleNextToken(last_logits, temperature, top_p);

    generated_tokens.push_back(next_token);

    if (next_token == tokenizer_->getEosTokenId()) {
      break;
    }

    std::string chunk = tokenizer_->decodeToken(next_token);
    if (callback && !chunk.empty()) {
      callback(chunk);
    }

    current_input_ids.push_back(next_token);
    current_mask.push_back(1);
  }

  ResponseData resp;
  resp.content = tokenizer_->decode(generated_tokens);
  resp.model = model_path_;
  resp.finish_reason = "stop";
  resp.prompt_tokens = static_cast<int>(encoded.input_ids.size());
  resp.completion_tokens = static_cast<int>(generated_tokens.size());
  resp.total_tokens = resp.prompt_tokens + resp.completion_tokens;
  return resp;
}

std::vector<float> OnnxClient::computeMeanPooling(const TensorData &hidden_state,
                                                 const std::vector<int64_t> &attention_mask) const {
  if (hidden_state.shape.size() < 3) {
    return hidden_state.float_data;
  }

  int64_t seq_len = hidden_state.shape[1];
  int64_t hidden_dim = hidden_state.shape[2];
  std::vector<float> pooled(hidden_dim, 0.0f);
  float mask_sum = 0.0f;

  for (int64_t s = 0; s < seq_len; ++s) {
    float mask_val = (s < static_cast<int64_t>(attention_mask.size())) ? static_cast<float>(attention_mask[s]) : 1.0f;
    if (mask_val <= 0.0f) continue;

    mask_sum += mask_val;
    for (int64_t h = 0; h < hidden_dim; ++h) {
      size_t idx = static_cast<size_t>(s * hidden_dim + h);
      if (idx < hidden_state.float_data.size()) {
        pooled[h] += hidden_state.float_data[idx] * mask_val;
      }
    }
  }

  if (mask_sum > 0.0f) {
    for (auto &v : pooled) {
      v /= mask_sum;
    }
  }

  return pooled;
}

EmbeddingResponse OnnxClient::embed(const std::vector<std::string> &inputs,
                                    const EmbeddingParams &params) {
  if (!engine_ || !engine_->isLoaded()) {
    throw ConfigurationException("OnnxClient: model is not loaded: " + model_path_);
  }

  EmbeddingResponse resp;
  resp.model = params.model.empty() ? model_path_ : params.model;
  int total_prompt_tokens = 0;

  for (const auto &text : inputs) {
    TokenizedPrompt enc = tokenizer_->encode(text);
    total_prompt_tokens += static_cast<int>(enc.input_ids.size());

    TensorShape shape = {1, static_cast<int64_t>(enc.input_ids.size())};
    OnnxTensorMap onnx_inputs;
    onnx_inputs["input_ids"] = TensorData::createInt64(shape, enc.input_ids);
    onnx_inputs["attention_mask"] = TensorData::createInt64(shape, enc.attention_mask);

    OnnxTensorMap outputs = engine_->run(onnx_inputs);

    if (outputs.find("last_hidden_state") != outputs.end()) {
      resp.embeddings.push_back(computeMeanPooling(outputs.at("last_hidden_state"), enc.attention_mask));
    } else if (outputs.find("embeddings") != outputs.end()) {
      resp.embeddings.push_back(outputs.at("embeddings").float_data);
    } else if (!outputs.empty()) {
      // 첫 번째 출력 텐서 활용
      resp.embeddings.push_back(outputs.begin()->second.float_data);
    } else {
      throw ParseException("OnnxClient: no embedding tensor produced by model");
    }
  }

  resp.prompt_tokens = total_prompt_tokens;
  resp.total_tokens = total_prompt_tokens;
  return resp;
}

} // namespace llm_client
