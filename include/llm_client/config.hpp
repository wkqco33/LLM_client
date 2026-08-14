#pragma once

#include <cstdlib>
#include <string>

namespace llm_client {

class Config {
public:
  // 환경 변수에서 값을 읽어오는 유틸리티 (ROS2 환경에서 유용함)
  static std::string getEnv(const std::string &key,
                            const std::string &default_value = "") {
    const char *val = std::getenv(key.c_str());
    if (val == nullptr) {
      return default_value;
    }
    return std::string(val);
  }
};

} // namespace llm_client
