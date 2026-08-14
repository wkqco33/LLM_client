#include "llm_client/ollama_client.hpp"
#include "llm_client/exceptions.hpp"
#include "llm_client/http_util.hpp"
#include "llm_client/logger.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace llm_client {

namespace {

json buildRequestBody(const std::vector<Message> &messages,
                      const RequestParams &params, const std::string &model,
                      bool streaming) {
  json j_req;
  j_req["model"] = model;
  j_req["stream"] = streaming;
  j_req["messages"] = detail::build_role_content_messages(messages);

  json options;
  detail::set_if_present(options, "temperature", params.temperature);
  detail::set_if_present(options, "top_p", params.top_p);
  detail::set_if_present(options, "num_predict", params.max_tokens);
  if (params.thinking.has_value()) {
    options["think"] = params.thinking.value();
    j_req["think"] = params.thinking.value();
  }
  if (!options.empty()) {
    j_req["options"] = options;
  }
  return j_req;
}

} // namespace

OllamaClient::OllamaClient(const std::string &base_url,
                           const std::string &api_key)
    : base_url_(base_url.empty() ? "http://localhost:11434" : base_url),
      api_key_(api_key) {}

ResponseData OllamaClient::chat(const std::vector<Message> &messages,
                                const RequestParams &params) {
  std::string model = params.model.empty() ? "llama3" : params.model;
  LLM_LOG_INFO("OllamaClient: chat request started with model: {}", model);

  json j_req = buildRequestBody(messages, params, model, /*streaming=*/false);

  std::string endpoint = base_url_ + "/api/chat";
  cpr::Header headers{{"Content-Type", "application/json"}};
  if (!api_key_.empty()) {
    headers["Authorization"] = "Bearer " + api_key_;
  }

  // HTTP POST 및 공통 예외/로깅 처리
  json j_res = detail::http_post_json(endpoint, headers, j_req, "Ollama",
                                      params.timeout_ms.value_or(60000),
                                      params.max_retries.value_or(0));

  ResponseData response_data;

  try {
    if (j_res.contains("message") && j_res["message"].is_object()) {
      const auto &msg = j_res["message"];
      if (msg.contains("content")) {
        response_data.content = msg["content"].get<std::string>();
      }
      if (msg.contains("thinking")) {
        response_data.reasoning_content = msg["thinking"].get<std::string>();
      } else if (msg.contains("reasoning_content")) {
        response_data.reasoning_content =
            msg["reasoning_content"].get<std::string>();
      }
    }

    response_data.model = j_res.value("model", model);
    response_data.finish_reason = j_res.value("done_reason", "stop");

    response_data.prompt_tokens = j_res.value("prompt_eval_count", 0);
    response_data.completion_tokens = j_res.value("eval_count", 0);
    response_data.total_tokens =
        response_data.prompt_tokens + response_data.completion_tokens;
  } catch (const std::exception &e) {
    std::string err_msg =
        "Ollama API response JSON parsing error: " + std::string(e.what());
    LLM_LOG_ERROR(err_msg);
    throw ParseException(err_msg);
  }

  LLM_LOG_INFO("OllamaClient: chat request completed (total tokens: {})",
               response_data.total_tokens);

  return response_data;
}

ResponseData OllamaClient::chatStream(const std::vector<Message> &messages,
                                      StreamCallback callback,
                                      const RequestParams &params) {
  std::string model = params.model.empty() ? "llama3" : params.model;
  LLM_LOG_INFO("OllamaClient: chatStream request started with model: {}", model);

  json j_req = buildRequestBody(messages, params, model, /*streaming=*/true);

  std::string endpoint = base_url_ + "/api/chat";
  cpr::Header headers{{"Content-Type", "application/json"}};
  if (!api_key_.empty()) {
    headers["Authorization"] = "Bearer " + api_key_;
  }

  ResponseData response_data;
  response_data.model = model;

  detail::http_post_stream(
      endpoint, headers, j_req, "Ollama",
      [&response_data, &callback](const std::string &line) {
        detail::parse_stream_json_line(
            line, "OllamaClient",
            [&](const json &j) {
              if (j.contains("message") && j["message"].is_object()) {
                const auto &msg = j["message"];
                if (msg.contains("content")) {
                  std::string chunk = msg["content"].get<std::string>();
                  response_data.content += chunk;
                  if (callback) {
                    callback(chunk);
                  }
                }
                if (msg.contains("thinking")) {
                  response_data.reasoning_content +=
                      msg["thinking"].get<std::string>();
                } else if (msg.contains("reasoning_content")) {
                  response_data.reasoning_content +=
                      msg["reasoning_content"].get<std::string>();
                }
              }
              if (j.value("done", false)) {
                response_data.finish_reason = j.value("done_reason", "stop");
                response_data.prompt_tokens = j.value("prompt_eval_count", 0);
                response_data.completion_tokens = j.value("eval_count", 0);
                response_data.total_tokens = response_data.prompt_tokens +
                                             response_data.completion_tokens;
              }
            },
            /*data_prefix=*/"");
      },
      params.timeout_ms.value_or(120000), params.max_retries.value_or(0));

  return response_data;
}

EmbeddingResponse OllamaClient::embed(const std::vector<std::string> &inputs,
                                      const EmbeddingParams &params) {
  std::string model =
      params.model.empty() ? "nomic-embed-text" : params.model;
  LLM_LOG_INFO("OllamaClient: embed request started with model: {} ({} inputs)",
               model, inputs.size());

  json j_req;
  j_req["model"] = model;
  j_req["input"] = inputs;

  std::string endpoint = base_url_ + "/api/embed";
  cpr::Header headers{{"Content-Type", "application/json"}};
  if (!api_key_.empty()) {
    headers["Authorization"] = "Bearer " + api_key_;
  }

  json j_res = detail::http_post_json(endpoint, headers, j_req, "Ollama",
                                      params.timeout_ms.value_or(60000),
                                      params.max_retries.value_or(0));

  EmbeddingResponse embedding_data;
  try {
    if (j_res.contains("embeddings") && j_res["embeddings"].is_array()) {
      for (const auto &item : j_res["embeddings"]) {
        embedding_data.embeddings.push_back(item.get<std::vector<float>>());
      }
    }

    embedding_data.model = j_res.value("model", model);
    embedding_data.prompt_tokens = j_res.value("prompt_eval_count", 0);
    embedding_data.total_tokens = embedding_data.prompt_tokens;
  } catch (const std::exception &e) {
    std::string err_msg =
        "Ollama API response JSON parsing error: " + std::string(e.what());
    LLM_LOG_ERROR(err_msg);
    throw ParseException(err_msg);
  }

  LLM_LOG_INFO("OllamaClient: embed request completed ({} embeddings)",
               embedding_data.embeddings.size());

  return embedding_data;
}

} // namespace llm_client
