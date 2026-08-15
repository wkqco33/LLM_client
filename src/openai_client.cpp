#include "llm_client/openai_client.hpp"
#include "llm_client/cpr_http_client.hpp"
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
  if (streaming) {
    j_req["stream"] = true;
  }
  j_req["messages"] = detail::build_role_content_messages(messages);
  detail::set_if_present(j_req, "temperature", params.temperature);
  detail::set_if_present(j_req, "max_tokens", params.max_tokens);
  detail::set_if_present(j_req, "top_p", params.top_p);

  if (params.response_format.has_value()) {
    if (*params.response_format == "json_object") {
      j_req["response_format"] = {{"type", "json_object"}};
    } else if (*params.response_format == "json_schema" && params.json_schema.has_value()) {
      try {
        j_req["response_format"] = {
            {"type", "json_schema"},
            {"json_schema", json::parse(*params.json_schema)}
        };
      } catch (const std::exception &) {
        j_req["response_format"] = {{"type", "json_object"}};
      }
    }
  }

  return j_req;
}

} // namespace

OpenAIClient::OpenAIClient(const std::string &api_key,
                           const std::string &base_url,
                           std::shared_ptr<IHttpClient> http_client)
    : api_key_(api_key), base_url_(base_url),
      http_client_(http_client ? std::move(http_client)
                               : std::make_shared<CprHttpClient>()) {}

ResponseData OpenAIClient::chat(const std::vector<Message> &messages,
                                const RequestParams &params) {
  std::string model = params.model.empty() ? "gpt-4o" : params.model;
  LLM_LOG_INFO("OpenAIClient: chat request started with model: {}", model);

  json j_req = buildRequestBody(messages, params, model, /*streaming=*/false);

  std::string endpoint = base_url_ + "/chat/completions";
  std::map<std::string, std::string> headers{
      {"Authorization", "Bearer " + api_key_},
      {"Content-Type", "application/json"}};

  json j_res = detail::http_post_json(*http_client_, endpoint, headers, j_req,
                                      "OpenAI",
                                      params.timeout_ms.value_or(30000),
                                      params.max_retries.value_or(0));

  ResponseData response_data;
  try {
    if (j_res.contains("choices") && j_res["choices"].is_array() &&
        !j_res["choices"].empty()) {
      const auto &choice = j_res["choices"][0];
      if (choice.contains("message") && choice["message"].contains("content")) {
        response_data.content = choice["message"]["content"].get<std::string>();
      }
      response_data.finish_reason = choice.value("finish_reason", "unknown");
    }

    response_data.model = j_res.value("model", model);

    if (j_res.contains("usage") && j_res["usage"].is_object()) {
      const auto &usage = j_res["usage"];
      response_data.prompt_tokens = usage.value("prompt_tokens", 0);
      response_data.completion_tokens = usage.value("completion_tokens", 0);
      response_data.total_tokens = usage.value("total_tokens", 0);
    }
  } catch (const std::exception &e) {
    std::string err_msg =
        "OpenAI API response JSON parsing error: " + std::string(e.what());
    LLM_LOG_ERROR(err_msg);
    throw ParseException(err_msg);
  }

  LLM_LOG_INFO("OpenAIClient: chat request completed (total tokens: {})",
               response_data.total_tokens);

  return response_data;
}

ResponseData OpenAIClient::chatStream(const std::vector<Message> &messages,
                                      StreamCallback callback,
                                      const RequestParams &params) {
  std::string model = params.model.empty() ? "gpt-4o" : params.model;
  LLM_LOG_INFO("OpenAIClient: chatStream request started with model: {}", model);

  json j_req = buildRequestBody(messages, params, model, /*streaming=*/true);

  std::string endpoint = base_url_ + "/chat/completions";
  std::map<std::string, std::string> headers{
      {"Authorization", "Bearer " + api_key_},
      {"Content-Type", "application/json"}};

  ResponseData response_data;
  response_data.model = model;

  detail::http_post_stream(
      *http_client_, endpoint, headers, j_req, "OpenAI",
      [&response_data, &callback](const std::string &line) {
        detail::parse_stream_json_line(line, "OpenAIClient", [&](const json &j) {
          if (j.contains("choices") && j["choices"].is_array() &&
              !j["choices"].empty()) {
            const auto &choice = j["choices"][0];
            if (choice.contains("delta") &&
                choice["delta"].contains("content")) {
              std::string chunk = choice["delta"]["content"].get<std::string>();
              response_data.content += chunk;
              if (callback) {
                callback(chunk);
              }
            }
            if (choice.contains("finish_reason") &&
                !choice["finish_reason"].is_null()) {
              response_data.finish_reason =
                  choice["finish_reason"].get<std::string>();
            }
          }
        });
      },
      params.timeout_ms.value_or(60000), params.max_retries.value_or(0));

  return response_data;
}

EmbeddingResponse OpenAIClient::embed(const std::vector<std::string> &inputs,
                                      const EmbeddingParams &params) {
  std::string model =
      params.model.empty() ? "text-embedding-3-small" : params.model;
  LLM_LOG_INFO("OpenAIClient: embed request started with model: {} ({} inputs)",
               model, inputs.size());

  json j_req;
  j_req["model"] = model;
  j_req["input"] = inputs;

  std::string endpoint = base_url_ + "/embeddings";
  std::map<std::string, std::string> headers{
      {"Authorization", "Bearer " + api_key_},
      {"Content-Type", "application/json"}};

  json j_res = detail::http_post_json(*http_client_, endpoint, headers, j_req,
                                      "OpenAI",
                                      params.timeout_ms.value_or(30000),
                                      params.max_retries.value_or(0));

  EmbeddingResponse embedding_data;
  try {
    if (j_res.contains("data") && j_res["data"].is_array()) {
      embedding_data.embeddings.resize(j_res["data"].size());
      for (const auto &item : j_res["data"]) {
        size_t index = item.value("index", size_t{0});
        if (index < embedding_data.embeddings.size() &&
            item.contains("embedding")) {
          embedding_data.embeddings[index] =
              item["embedding"].get<std::vector<float>>();
        }
      }
    }

    embedding_data.model = j_res.value("model", model);

    if (j_res.contains("usage") && j_res["usage"].is_object()) {
      const auto &usage = j_res["usage"];
      embedding_data.prompt_tokens = usage.value("prompt_tokens", 0);
      embedding_data.total_tokens = usage.value("total_tokens", 0);
    }
  } catch (const std::exception &e) {
    std::string err_msg =
        "OpenAI API response JSON parsing error: " + std::string(e.what());
    LLM_LOG_ERROR(err_msg);
    throw ParseException(err_msg);
  }

  LLM_LOG_INFO("OpenAIClient: embed request completed (total tokens: {})",
               embedding_data.total_tokens);

  return embedding_data;
}

} // namespace llm_client
