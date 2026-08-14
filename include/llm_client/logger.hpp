#pragma once

#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace llm_client {

class Logger {
public:
  // Header-only singleton 팩토리 (초기화 보장)
  static std::shared_ptr<spdlog::logger> get() {
    static std::shared_ptr<spdlog::logger> logger = []() {
      auto console_sink =
          std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      // 포맷 통일: [YYYY-MM-DD HH:MM:SS] [주황/기타 컬러 LEVEL] 메시지
      console_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");

      auto l = std::make_shared<spdlog::logger>("llm_client", console_sink);
      spdlog::register_logger(l);
      l->set_level(spdlog::level::info); // 기본 수준 Info로 설정
      return l;
    }();
    return logger;
  }
};

} // namespace llm_client

// 유틸성 매크로: 프로젝트 전역에서 짧은 줄임말로 쉽게 작성
#define LLM_LOG_TRACE(...) ::llm_client::Logger::get()->trace(__VA_ARGS__)
#define LLM_LOG_DEBUG(...) ::llm_client::Logger::get()->debug(__VA_ARGS__)
#define LLM_LOG_INFO(...) ::llm_client::Logger::get()->info(__VA_ARGS__)
#define LLM_LOG_WARN(...) ::llm_client::Logger::get()->warn(__VA_ARGS__)
#define LLM_LOG_ERROR(...) ::llm_client::Logger::get()->error(__VA_ARGS__)
#define LLM_LOG_CRITICAL(...) ::llm_client::Logger::get()->critical(__VA_ARGS__)
