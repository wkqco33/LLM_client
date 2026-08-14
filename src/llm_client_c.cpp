#include "llm_client/llm_client_c.h"
#include "llm_client/exceptions.hpp"
#include "llm_client/llm_client_factory.hpp"
#include "llm_client/llm_client_interface.hpp"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static char *c_strdup(const std::string &str) {
  if (str.empty()) {
    char *empty = static_cast<char *>(std::malloc(1));
    if (empty) {
      empty[0] = '\0';
    }
    return empty;
  }
  char *copy = static_cast<char *>(std::malloc(str.size() + 1));
  if (copy) {
    std::memcpy(copy, str.c_str(), str.size() + 1);
  }
  return copy;
}

static llm_client::RequestParams
convert_params(const llm_request_params_t *params) {
  llm_client::RequestParams p;
  if (!params) {
    return p;
  }

  if (params->model) {
    p.model = params->model;
  }
  if (params->has_temperature) {
    p.temperature = params->temperature;
  }
  if (params->has_top_p) {
    p.top_p = params->top_p;
  }
  if (params->has_max_tokens) {
    p.max_tokens = params->max_tokens;
  }
  if (params->has_thinking) {
    p.thinking = params->thinking;
  }
  if (params->timeout_ms > 0) {
    p.timeout_ms = params->timeout_ms;
  }
  if (params->max_retries > 0) {
    p.max_retries = params->max_retries;
  }

  return p;
}

static std::vector<llm_client::Message>
convert_messages(const llm_message_t *messages, size_t count) {
  std::vector<llm_client::Message> msgs;
  if (!messages || count == 0) {
    return msgs;
  }
  msgs.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    msgs.push_back({messages[i].role ? messages[i].role : "",
                    messages[i].content ? messages[i].content : ""});
  }
  return msgs;
}

static llm_client::EmbeddingParams
convert_embedding_params(const llm_embedding_params_t *params) {
  llm_client::EmbeddingParams p;
  if (!params) {
    return p;
  }

  if (params->model) {
    p.model = params->model;
  }
  if (params->timeout_ms > 0) {
    p.timeout_ms = params->timeout_ms;
  }
  if (params->max_retries > 0) {
    p.max_retries = params->max_retries;
  }

  return p;
}

static std::vector<std::string> convert_inputs(const char **inputs,
                                                size_t count) {
  std::vector<std::string> out;
  if (!inputs || count == 0) {
    return out;
  }
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.emplace_back(inputs[i] ? inputs[i] : "");
  }
  return out;
}

static llm_response_data_t
convert_response(const llm_client::ResponseData &res) {
  llm_response_data_t out;
  std::memset(&out, 0, sizeof(out));
  out.content = c_strdup(res.content);
  out.reasoning_content = c_strdup(res.reasoning_content);
  out.model = c_strdup(res.model);
  out.finish_reason = c_strdup(res.finish_reason);
  out.prompt_tokens = res.prompt_tokens;
  out.completion_tokens = res.completion_tokens;
  out.total_tokens = res.total_tokens;
  out.status_code = 0;
  out.error_message = nullptr;
  return out;
}

static llm_response_data_t make_error_response(int status_code,
                                                const std::string &err) {
  llm_response_data_t out;
  std::memset(&out, 0, sizeof(out));
  out.status_code = status_code;
  out.error_message = c_strdup(err);
  out.content = c_strdup("");
  out.reasoning_content = c_strdup("");
  out.model = c_strdup("");
  out.finish_reason = c_strdup("error");
  return out;
}

static llm_embedding_response_t
convert_embedding_response(const llm_client::EmbeddingResponse &res) {
  llm_embedding_response_t out;
  std::memset(&out, 0, sizeof(out));

  out.count = res.embeddings.size();
  out.model = c_strdup(res.model);
  out.prompt_tokens = res.prompt_tokens;
  out.total_tokens = res.total_tokens;
  out.status_code = 0;
  out.error_message = nullptr;

  if (out.count == 0) {
    out.embeddings = nullptr;
    out.embedding_dims = nullptr;
    return out;
  }

  out.embeddings = static_cast<float **>(std::malloc(out.count * sizeof(float *)));
  out.embedding_dims = static_cast<size_t *>(std::malloc(out.count * sizeof(size_t)));
  for (size_t i = 0; i < out.count; ++i) {
    const auto &vec = res.embeddings[i];
    out.embedding_dims[i] = vec.size();
    if (vec.empty()) {
      out.embeddings[i] = nullptr;
      continue;
    }
    out.embeddings[i] = static_cast<float *>(std::malloc(vec.size() * sizeof(float)));
    std::memcpy(out.embeddings[i], vec.data(), vec.size() * sizeof(float));
  }

  return out;
}

static llm_embedding_response_t
make_error_embedding_response(int status_code, const std::string &err) {
  llm_embedding_response_t out;
  std::memset(&out, 0, sizeof(out));
  out.status_code = status_code;
  out.error_message = c_strdup(err);
  out.model = c_strdup("");
  return out;
}

// llm_client_embed가 사용하는 예외->에러코드 변환 (safe_invoke와 동일한
// 예외 5종 매핑을 EmbeddingResponse 반환 함수에 적용).
template <typename Fn>
static llm_embedding_response_t safe_invoke_embed(Fn &&fn) {
  try {
    return convert_embedding_response(fn());
  } catch (const llm_client::NetworkException &e) {
    return make_error_embedding_response(1, e.what());
  } catch (const llm_client::APIException &e) {
    return make_error_embedding_response(2, e.what());
  } catch (const llm_client::ParseException &e) {
    return make_error_embedding_response(3, e.what());
  } catch (const llm_client::ConfigurationException &e) {
    return make_error_embedding_response(4, e.what());
  } catch (const std::exception &e) {
    return make_error_embedding_response(5, e.what());
  }
}

// llm_client_chat/llm_client_chat_stream이 공통으로 쓰는 예외->에러코드 변환.
// fn()이 llm_client::ResponseData를 반환하는 호출(chat/chatStream)을 감싸서,
// 던져질 수 있는 예외 5종을 각각 다른 status_code로 변환한다.
template <typename Fn>
static llm_response_data_t safe_invoke(Fn &&fn) {
  try {
    return convert_response(fn());
  } catch (const llm_client::NetworkException &e) {
    return make_error_response(1, e.what());
  } catch (const llm_client::APIException &e) {
    return make_error_response(2, e.what());
  } catch (const llm_client::ParseException &e) {
    return make_error_response(3, e.what());
  } catch (const llm_client::ConfigurationException &e) {
    return make_error_response(4, e.what());
  } catch (const std::exception &e) {
    return make_error_response(5, e.what());
  }
}

extern "C" {

llm_client_h llm_client_create(const char *provider, const char *api_key,
                                const char *base_url,
                                const char *api_version) {
  if (!provider) {
    return nullptr;
  }
  try {
    std::string prov = provider;
    std::string key = api_key ? api_key : "";
    std::string url = base_url ? base_url : "";
    std::string ver = api_version ? api_version : "";

    auto client = llm_client::LLMClientFactory::create(prov, key, url, ver);
    return static_cast<llm_client_h>(client.release());
  } catch (...) {
    return nullptr;
  }
}

void llm_client_destroy(llm_client_h client) {
  if (client) {
    delete static_cast<llm_client::LLMClientInterface *>(client);
  }
}

llm_response_data_t
llm_client_chat(llm_client_h client, const llm_message_t *messages,
                size_t message_count, const llm_request_params_t *params) {
  if (!client) {
    return make_error_response(4, "Invalid client handle");
  }

  auto *cpp_client = static_cast<llm_client::LLMClientInterface *>(client);
  auto msgs = convert_messages(messages, message_count);
  auto req_params = convert_params(params);

  return safe_invoke([&] { return cpp_client->chat(msgs, req_params); });
}

llm_response_data_t
llm_client_generate(llm_client_h client, const char *prompt,
                    const llm_request_params_t *params) {
  llm_message_t msg;
  msg.role = "user";
  msg.content = prompt ? prompt : "";
  return llm_client_chat(client, &msg, 1, params);
}

llm_response_data_t llm_client_chat_stream(
    llm_client_h client, const llm_message_t *messages, size_t message_count,
    llm_stream_callback_t callback, void *user_data,
    const llm_request_params_t *params) {
  if (!client) {
    return make_error_response(4, "Invalid client handle");
  }

  auto *cpp_client = static_cast<llm_client::LLMClientInterface *>(client);
  auto msgs = convert_messages(messages, message_count);
  auto req_params = convert_params(params);

  auto cpp_cb = [callback, user_data](const std::string &chunk) {
    if (callback) {
      callback(chunk.c_str(), user_data);
    }
  };

  return safe_invoke(
      [&] { return cpp_client->chatStream(msgs, cpp_cb, req_params); });
}

void llm_response_free(llm_response_data_t *response) {
  if (!response) {
    return;
  }
  if (response->content) {
    std::free(response->content);
    response->content = nullptr;
  }
  if (response->reasoning_content) {
    std::free(response->reasoning_content);
    response->reasoning_content = nullptr;
  }
  if (response->model) {
    std::free(response->model);
    response->model = nullptr;
  }
  if (response->finish_reason) {
    std::free(response->finish_reason);
    response->finish_reason = nullptr;
  }
  if (response->error_message) {
    std::free(response->error_message);
    response->error_message = nullptr;
  }
}

llm_embedding_response_t
llm_client_embed(llm_client_h client, const char **inputs, size_t input_count,
                 const llm_embedding_params_t *params) {
  if (!client) {
    return make_error_embedding_response(4, "Invalid client handle");
  }

  auto *cpp_client = static_cast<llm_client::LLMClientInterface *>(client);
  auto cpp_inputs = convert_inputs(inputs, input_count);
  auto req_params = convert_embedding_params(params);

  return safe_invoke_embed(
      [&] { return cpp_client->embed(cpp_inputs, req_params); });
}

void llm_embedding_response_free(llm_embedding_response_t *response) {
  if (!response) {
    return;
  }
  if (response->embeddings) {
    for (size_t i = 0; i < response->count; ++i) {
      if (response->embeddings[i]) {
        std::free(response->embeddings[i]);
      }
    }
    std::free(response->embeddings);
    response->embeddings = nullptr;
  }
  if (response->embedding_dims) {
    std::free(response->embedding_dims);
    response->embedding_dims = nullptr;
  }
  if (response->model) {
    std::free(response->model);
    response->model = nullptr;
  }
  if (response->error_message) {
    std::free(response->error_message);
    response->error_message = nullptr;
  }
}

} // extern "C"
