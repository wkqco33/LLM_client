#include "llm_client/llm_client_c.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

TEST(CABITest, CreateAndDestroyClient) {
  llm_client_h client_openai = llm_client_create("openai", "mock-key", "https://api.openai.com/v1", "");
  ASSERT_NE(client_openai, nullptr);
  llm_client_destroy(client_openai);

  llm_client_h client_ollama = llm_client_create("ollama", "", "http://localhost:11434", "");
  ASSERT_NE(client_ollama, nullptr);
  llm_client_destroy(client_ollama);

  llm_client_h client_onnx = llm_client_create("onnx", "", "/path/to/model.onnx", "/path/to/tokenizer.json");
  ASSERT_NE(client_onnx, nullptr);
  llm_client_destroy(client_onnx);

  llm_client_h client_onnx_empty = llm_client_create("onnx", "", "", "");
  EXPECT_EQ(client_onnx_empty, nullptr);

  llm_client_h client_invalid = llm_client_create("non_existent_provider", "", "", "");
  EXPECT_EQ(client_invalid, nullptr);

  llm_client_h client_null = llm_client_create(nullptr, nullptr, nullptr, nullptr);
  EXPECT_EQ(client_null, nullptr);
}

TEST(CABITest, NullClientHandling) {
  llm_message_t msg{"user", "Hello"};
  llm_response_data_t res = llm_client_chat(nullptr, &msg, 1, nullptr);
  EXPECT_EQ(res.status_code, 4); // ConfigurationException / Invalid client handle
  ASSERT_NE(res.error_message, nullptr);
  EXPECT_STREQ(res.error_message, "Invalid client handle");
  llm_response_free(&res);

  llm_response_data_t gen_res = llm_client_generate(nullptr, "Hello", nullptr);
  EXPECT_EQ(gen_res.status_code, 4);
  llm_response_free(&gen_res);

  llm_response_data_t stream_res = llm_client_chat_stream(nullptr, &msg, 1, nullptr, nullptr, nullptr);
  EXPECT_EQ(stream_res.status_code, 4);
  llm_response_free(&stream_res);

  const char *inputs[] = {"test"};
  llm_embedding_response_t embed_res = llm_client_embed(nullptr, inputs, 1, nullptr);
  EXPECT_EQ(embed_res.status_code, 4);
  llm_embedding_response_free(&embed_res);
}

TEST(CABITest, ResponseFree_NullSafety) {
  // null 포인터를 넘겨도 crash되지 않아야 함
  llm_response_free(nullptr);
  llm_embedding_response_free(nullptr);
}

TEST(CABITest, ParamsConversionAndErrorMapping) {
  // 실제 연결 불가능한 로컬 주소로 요청하여 Network Exception (status_code: 1) 매핑 검증
  llm_client_h client = llm_client_create("openai", "test", "http://127.0.0.1:59999", "");
  ASSERT_NE(client, nullptr);

  llm_request_params_t params;
  params.model = "gpt-4o";
  params.has_temperature = true;
  params.temperature = 0.5f;
  params.has_top_p = true;
  params.top_p = 0.9f;
  params.has_max_tokens = true;
  params.max_tokens = 50;
  params.has_thinking = false;
  params.thinking = false;
  params.timeout_ms = 500;
  params.max_retries = 0;

  llm_response_data_t res = llm_client_generate(client, "Hello", &params);
  // 연결 실패이므로 status_code는 1 (NetworkException)이어야 함
  EXPECT_EQ(res.status_code, 1);
  EXPECT_STREQ(res.finish_reason, "error");
  ASSERT_NE(res.error_message, nullptr);
  EXPECT_STRNE(res.error_message, "");

  llm_response_free(&res);
  llm_client_destroy(client);
}
