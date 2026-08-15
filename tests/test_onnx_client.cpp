#include "llm_client/onnx_client.hpp"
#include "llm_client/exceptions.hpp"
#include "llm_client/mock_onnx_engine.hpp"
#include "llm_client/simple_tokenizer.hpp"
#include <gtest/gtest.h>
#include <numeric>

using namespace llm_client;

class OnnxClientTest : public ::testing::Test {
protected:
  std::shared_ptr<MockOnnxEngine> mock_engine;
  std::shared_ptr<SimpleTokenizer> tokenizer;

  void SetUp() override {
    mock_engine = std::make_shared<MockOnnxEngine>();
    tokenizer = std::make_shared<SimpleTokenizer>();

    // 기본 어휘 사전 등록
    tokenizer->addToken("<pad>", 0);
    tokenizer->addToken("<s>", 1);
    tokenizer->addToken("</s>", 2); // EOS
    tokenizer->addToken("<unk>", 3);
    tokenizer->addToken("Hello", 4);
    tokenizer->addToken("world", 5);
    tokenizer->addToken("!", 6);
    tokenizer->addToken("AI", 7);
    tokenizer->addToken("assistant", 8);
    tokenizer->addToken("ONNX", 9);
    tokenizer->addToken("rocks", 10);
    tokenizer->setSpecialTokens(1, 2, 0, 3);
  }

  // 지정한 token_id의 logit이 가장 높은 텐서 생성 (어휘 크기 12 기준)
  TensorData makeLogitsTensor(int64_t winner_token_id, size_t vocab_size = 12) {
    std::vector<float> logits(vocab_size, -10.0f);
    if (winner_token_id < static_cast<int64_t>(vocab_size)) {
      logits[winner_token_id] = 10.0f; // 최댓값
    }
    // shape: [1, 1, vocab_size]
    return TensorData::createFloat({1, 1, static_cast<int64_t>(vocab_size)}, logits);
  }
};

TEST_F(OnnxClientTest, Tokenizer_EncodeDecode) {
  TokenizedPrompt enc = tokenizer->encode("Hello world !");
  ASSERT_EQ(enc.input_ids.size(), 3);
  EXPECT_EQ(enc.input_ids[0], 4); // "Hello"
  EXPECT_EQ(enc.input_ids[1], 5); // "world"
  EXPECT_EQ(enc.input_ids[2], 6); // "!"

  std::string dec = tokenizer->decode(enc.input_ids);
  EXPECT_EQ(dec, "Hello world !");
}

TEST_F(OnnxClientTest, ChatSuccess_GeneratesTextUsingLogits) {
  // 토큰 시퀀스: "ONNX" (9) -> "rocks" (10) -> "</s>" (2, EOS)
  mock_engine->queueOutput({{"logits", makeLogitsTensor(9)}});  // 1번째 토큰: ONNX
  mock_engine->queueOutput({{"logits", makeLogitsTensor(10)}}); // 2번째 토큰: rocks
  mock_engine->queueOutput({{"logits", makeLogitsTensor(2)}});  // 3번째 토큰: </s> (종료)

  OnnxClient client("/path/to/model.onnx", "", mock_engine, tokenizer);

  RequestParams params;
  params.max_tokens = 10;
  ResponseData res = client.generate("Hello", params);

  EXPECT_EQ(res.content, "ONNX rocks");
  EXPECT_EQ(res.finish_reason, "stop");
  EXPECT_EQ(res.completion_tokens, 3);
  EXPECT_EQ(mock_engine->getRunCount(), 3);

  // 첫 번째 실행 시 입력 텐서 확인
  const auto &first_input = mock_engine->getRecordedInputs()[0];
  EXPECT_TRUE(first_input.find("input_ids") != first_input.end());
  EXPECT_TRUE(first_input.find("attention_mask") != first_input.end());
}

TEST_F(OnnxClientTest, ChatStream_EmitsTokensInRealtime) {
  mock_engine->queueOutput({{"logits", makeLogitsTensor(7)}});  // "AI"
  mock_engine->queueOutput({{"logits", makeLogitsTensor(8)}});  // "assistant"
  mock_engine->queueOutput({{"logits", makeLogitsTensor(2)}});  // EOS

  OnnxClient client("/path/to/model.onnx", "", mock_engine, tokenizer);

  std::vector<std::string> chunks;
  ResponseData res = client.generateStream("Prompt", [&](const std::string &chunk) {
    chunks.push_back(chunk);
  });

  ASSERT_EQ(chunks.size(), 2);
  EXPECT_EQ(chunks[0], "AI");
  EXPECT_EQ(chunks[1], "assistant");
  EXPECT_EQ(res.content, "AI assistant");
  EXPECT_EQ(res.finish_reason, "stop");
}

TEST_F(OnnxClientTest, EmbedSuccess_ComputesMeanPooling) {
  // 3개 토큰, 임베딩 차원 4: shape [1, 3, 4]
  // 토큰 0: [1.0, 1.0, 1.0, 1.0]
  // 토큰 1: [2.0, 2.0, 2.0, 2.0]
  // 토큰 2: [3.0, 3.0, 3.0, 3.0]
  // mean pooling 결과: [2.0, 2.0, 2.0, 2.0]
  std::vector<float> hidden_data = {
      1.0f, 1.0f, 1.0f, 1.0f,
      2.0f, 2.0f, 2.0f, 2.0f,
      3.0f, 3.0f, 3.0f, 3.0f
  };
  TensorData last_hidden_state = TensorData::createFloat({1, 3, 4}, hidden_data);
  mock_engine->setDefaultOutput({{"last_hidden_state", last_hidden_state}});

  OnnxClient client("/path/to/embed_model.onnx", "", mock_engine, tokenizer);

  EmbeddingResponse res = client.embed({"Hello world !"});
  ASSERT_EQ(res.embeddings.size(), 1);
  ASSERT_EQ(res.embeddings[0].size(), 4);
  EXPECT_FLOAT_EQ(res.embeddings[0][0], 2.0f);
  EXPECT_FLOAT_EQ(res.embeddings[0][1], 2.0f);
  EXPECT_FLOAT_EQ(res.embeddings[0][2], 2.0f);
  EXPECT_FLOAT_EQ(res.embeddings[0][3], 2.0f);
}

TEST_F(OnnxClientTest, UnloadedEngine_ThrowsConfigurationException) {
  mock_engine->setLoaded(false);

  OnnxClient client("/invalid/path.onnx", "", mock_engine, tokenizer);
  EXPECT_THROW(client.generate("Hello"), ConfigurationException);
  EXPECT_FALSE(client.validateConnection());
}
