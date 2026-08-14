#include "llm_client/types.hpp"
#include "llm_client/http_util.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "[Test Multimodal Schema]" << std::endl;

    llm_client::Message msg;
    msg.role = "user";
    msg.blocks.push_back(llm_client::ContentBlock::makeText("Describe this image"));
    msg.blocks.push_back(llm_client::ContentBlock::makeImageUrl("https://example.com/image.jpg"));
    msg.blocks.push_back(llm_client::ContentBlock::makeImageBase64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "image/png"));

    std::vector<llm_client::Message> messages = { msg };
    nlohmann::json j_messages = llm_client::detail::build_role_content_messages(messages);

    std::cout << "Serialized messages JSON:\n" << j_messages.dump(2) << std::endl;

    assert(j_messages.is_array());
    assert(j_messages[0]["role"] == "user");
    assert(j_messages[0]["content"].is_array());
    assert(j_messages[0]["content"].size() == 3);
    assert(j_messages[0]["content"][0]["type"] == "text");
    assert(j_messages[0]["content"][1]["type"] == "image_url");
    assert(j_messages[0]["content"][2]["image_url"]["url"].get<std::string>().rfind("data:image/png;base64,", 0) == 0);

    llm_client::RequestParams params;
    params.response_format = "json_object";
    params.json_schema = R"({"name": "user", "strict": true})";

    assert(params.response_format.value() == "json_object");
    assert(!params.json_schema.value().empty());

    std::cout << "Multimodal & Structured Output schema test passed!" << std::endl;
    return 0;
}
