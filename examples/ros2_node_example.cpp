/*
 * 이 코드는 ROS2 (rclcpp) 패키지 내에서 본 라이브러리를 사용하는 예시를
 * 보여주기 위한 더미 스켈레톤 코드입니다. ROS2 환경이 아니므로 빌드에는
 * 포함되지 않습니다. ROS2 pkg 내에 src 폴더에 넣고 활용하시면 됩니다.
 */

#if 0

#include "llm_client/config.hpp"
#include "llm_client/llm_client_factory.hpp"
#include "rclcpp/rclcpp.hpp"
#include <memory>

class LLMNode : public rclcpp::Node {
public:
    LLMNode() : Node("llm_service_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing LLM Node...");
        
        // 파라미터나 환경변수로부터 API Key 설정
        std::string provider = this->declare_parameter("llm_provider", "openai");
        std::string api_key = llm_client::Config::getEnv("LLM_API_KEY", "");
        
        if (api_key.empty()) {
            RCLCPP_ERROR(this->get_logger(), "LLM API KEY is empty! Set LLM_API_KEY env var.");
            return;
        }

        try {
            client_ = llm_client::LLMClientFactory::create(provider, api_key);
            RCLCPP_INFO(this->get_logger(), "[%s] Client Created successfully.", provider.c_str());
            
            // 테스트: 비동기 타이머나 ROS2 Service Callback 내부에서 다음과 같이 호출
            // auto response = client_->generate("ROS2 로봇 시스템을 제어하는 인공지능이 되어볼래?");
            // RCLCPP_INFO(this->get_logger(), "LLM Response: %s", response.content.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create LLM Client: %s", e.what());
        }
    }

private:
    std::unique_ptr<llm_client::LLMClientInterface> client_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LLMNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

#endif
