#include "llm_client/http_util.hpp"
#include "llm_client/types.hpp"
#include <gtest/gtest.h>

using namespace llm_client;

TEST(MultimodalTest, BuildRoleContentMessagesWithMultipleBlocks) {
  Message msg;
  msg.role = "user";
  msg.blocks.push_back(ContentBlock::makeText("Describe this image"));
  msg.blocks.push_back(ContentBlock::makeImageUrl("https://example.com/image.jpg"));
  msg.blocks.push_back(ContentBlock::makeImageBase64(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      "image/png"));

  std::vector<Message> messages = {msg};
  nlohmann::json j_messages = detail::build_role_content_messages(messages);

  ASSERT_TRUE(j_messages.is_array());
  ASSERT_EQ(j_messages.size(), 1);
  EXPECT_EQ(j_messages[0]["role"], "user");

  const auto &content = j_messages[0]["content"];
  ASSERT_TRUE(content.is_array());
  ASSERT_EQ(content.size(), 3);
  EXPECT_EQ(content[0]["type"], "text");
  EXPECT_EQ(content[0]["text"], "Describe this image");

  EXPECT_EQ(content[1]["type"], "image_url");
  EXPECT_EQ(content[1]["image_url"]["url"], "https://example.com/image.jpg");

  EXPECT_EQ(content[2]["type"], "image_url");
  std::string data_url = content[2]["image_url"]["url"].get<std::string>();
  EXPECT_EQ(data_url.rfind("data:image/png;base64,", 0), 0);
}

TEST(MultimodalTest, RequestParamsStructuredOutput) {
  RequestParams params;
  params.response_format = "json_object";
  params.json_schema = R"({"name": "user", "strict": true})";

  ASSERT_TRUE(params.response_format.has_value());
  EXPECT_EQ(params.response_format.value(), "json_object");
  ASSERT_TRUE(params.json_schema.has_value());
  EXPECT_FALSE(params.json_schema.value().empty());
}
