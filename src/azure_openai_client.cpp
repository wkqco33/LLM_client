#include "llm_client/azure_openai_client.hpp"
#include "llm_client/exceptions.hpp"
#include "llm_client/http_util.hpp"
#include "llm_client/logger.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace llm_client {

namespace {

// gpt-5, o1, o3, o4-mini 등 reasoning 계열 배포는 max_tokens 대신
// max_completion_tokens를 요구한다. 배포명은 리소스마다 자유롭게 지정되므로
// 완벽한 판별은 불가능하지만, 알려진 모델 계열의 접두사로 최대한 커버한다.
bool requiresMaxCompletionTokens(const std::string &deployment_name) {
  static const char *const kReasoningPrefixes[] = {"gpt-5", "o1", "o3",
                                                    "o4-mini", "o4"};
  for (const char *prefix : kReasoningPrefixes) {
    if (deployment_name.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

json buildRequestBody(const std::vector<Message> &messages,
                      const RequestParams &params,
                      const std::string &deployment_name, bool streaming) {
  json j_req;
  if (streaming) {
    j_req["stream"] = true;
  }
  j_req["messages"] = detail::build_role_content_messages(messages);

  detail::set_if_present(j_req, "temperature", params.temperature);
  detail::set_if_present(j_req, "top_p", params.top_p);
  if (params.max_tokens.has_value()) {
    // reasoning 계열(gpt-5, o1, o3, o4-mini 등)은 max_tokens 대신
    // max_completion_tokens를 요구합니다.
    const char *key = requiresMaxCompletionTokens(deployment_name)
                          ? "max_completion_tokens"
                          : "max_tokens";
    j_req[key] = params.max_tokens.value();
  }
  return j_req;
}

} // namespace

AzureOpenAIClient::AzureOpenAIClient(const std::string &api_key,
                                     const std::string &api_version,
                                     const std::string &base_url)
    : api_key_(api_key), api_version_(api_version), base_url_(base_url) {}

ResponseData AzureOpenAIClient::chat(const std::vector<Message> &messages,
                                     const RequestParams &params) {
  std::string deployment_name = params.model.empty() ? "gpt-4.1" : params.model;

  // params.model 은 Azure OpenAI에서 Deployment Name 역할로 사용될 수 있습니다.
  LLM_LOG_INFO("AzureOpenAIClient: chat request started with deployment: {}",
               deployment_name);

  json j_req = buildRequestBody(messages, params, deployment_name,
                                /*streaming=*/false);

  // Azure OpenAI Endpoint:
  // {base_url}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}
  std::string endpoint = base_url_ + "/openai/deployments/" + deployment_name +
                         "/chat/completions?api-version=" + api_version_;
  cpr::Header headers{{"api-key", api_key_},
                      {"Content-Type", "application/json"}};

  // HTTP POST 및 공통 예외/로깅 처리
  json j_res =
      detail::http_post_json(endpoint, headers, j_req, "Azure OpenAI",
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

    response_data.model = j_res.value("model", deployment_name);

    if (j_res.contains("usage") && j_res["usage"].is_object()) {
      const auto &usage = j_res["usage"];
      response_data.prompt_tokens = usage.value("prompt_tokens", 0);
      response_data.completion_tokens = usage.value("completion_tokens", 0);
      response_data.total_tokens = usage.value("total_tokens", 0);
    }
  } catch (const std::exception &e) {
    std::string err_msg = "Azure OpenAI API response JSON parsing error: " +
                          std::string(e.what());
    LLM_LOG_ERROR(err_msg);
    throw ParseException(err_msg);
  }

  LLM_LOG_INFO("AzureOpenAIClient: chat request completed (total tokens: {})",
               response_data.total_tokens);

  return response_data;
}

ResponseData AzureOpenAIClient::chatStream(const std::vector<Message> &messages,
                                           StreamCallback callback,
                                           const RequestParams &params) {
  std::string deployment_name =
      params.model.empty() ? "gpt-4.1" : params.model;
  LLM_LOG_INFO(
      "AzureOpenAIClient: chatStream request started with deployment: {}",
      deployment_name);

  json j_req = buildRequestBody(messages, params, deployment_name,
                                /*streaming=*/true);

  std::string endpoint = base_url_ + "/openai/deployments/" + deployment_name +
                         "/chat/completions?api-version=" + api_version_;
  cpr::Header headers{{"api-key", api_key_},
                      {"Content-Type", "application/json"}};

  ResponseData response_data;
  response_data.model = deployment_name;

  detail::http_post_stream(
      endpoint, headers, j_req, "Azure OpenAI",
      [&response_data, &callback](const std::string &line) {
        detail::parse_stream_json_line(
            line, "AzureOpenAIClient", [&](const json &j) {
              if (j.contains("choices") && j["choices"].is_array() &&
                  !j["choices"].empty()) {
                const auto &choice = j["choices"][0];
                if (choice.contains("delta") &&
                    choice["delta"].contains("content")) {
                  std::string chunk =
                      choice["delta"]["content"].get<std::string>();
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

EmbeddingResponse
AzureOpenAIClient::embed(const std::vector<std::string> &inputs,
                         const EmbeddingParams &params) {
  std::string deployment_name =
      params.model.empty() ? "text-embedding-3-small" : params.model;
  LLM_LOG_INFO(
      "AzureOpenAIClient: embed request started with deployment: {} ({} inputs)",
      deployment_name, inputs.size());

  json j_req;
  j_req["input"] = inputs;

  // Azure OpenAI Endpoint:
  // {base_url}/openai/deployments/{deployment_name}/embeddings?api-version={api_version}
  std::string endpoint = base_url_ + "/openai/deployments/" +
                         deployment_name +
                         "/embeddings?api-version=" + api_version_;
  cpr::Header headers{{"api-key", api_key_},
                      {"Content-Type", "application/json"}};

  json j_res =
      detail::http_post_json(endpoint, headers, j_req, "Azure OpenAI",
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

    embedding_data.model = j_res.value("model", deployment_name);

    if (j_res.contains("usage") && j_res["usage"].is_object()) {
      const auto &usage = j_res["usage"];
      embedding_data.prompt_tokens = usage.value("prompt_tokens", 0);
      embedding_data.total_tokens = usage.value("total_tokens", 0);
    }
  } catch (const std::exception &e) {
    std::string err_msg = "Azure OpenAI API response JSON parsing error: " +
                          std::string(e.what());
    LLM_LOG_ERROR(err_msg);
    throw ParseException(err_msg);
  }

  LLM_LOG_INFO("AzureOpenAIClient: embed request completed (total tokens: {})",
               embedding_data.total_tokens);

  return embedding_data;
}

} // namespace llm_client
