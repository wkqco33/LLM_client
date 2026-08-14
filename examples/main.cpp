#include "llm_client/config.hpp"
#include "llm_client/llm_client_factory.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <replxx.hxx>
#include <string>
#include <termcolor/termcolor.hpp>
#include <vector>

#ifndef LLM_CLIENT_ENV_FILE
#define LLM_CLIENT_ENV_FILE ""
#endif

namespace {

// 프로젝트 루트의 .env 파일을 런타임에 읽어 KEY=VALUE 항목을 프로세스
// 환경 변수로 로드한다. API 키를 컴파일 타임 매크로로 바이너리에 굽지
// 않기 위한 개발 편의 기능이며, 이미 설정된 환경 변수는 덮어쓰지 않는다.
void loadDotEnvFile(const std::string &path) {
  if (path.empty()) {
    return;
  }
  std::ifstream file(path);
  if (!file.is_open()) {
    return;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    auto eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      continue;
    }

    std::string key = line.substr(0, eq_pos);
    std::string value = line.substr(eq_pos + 1);

    auto trim = [](std::string &s) {
      size_t start = s.find_first_not_of(" \t\r\n");
      size_t end = s.find_last_not_of(" \t\r\n");
      s = (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    };
    trim(key);
    trim(value);

    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                              (value.front() == '\'' && value.back() == '\''))) {
      value = value.substr(1, value.size() - 2);
    }

    if (key.empty()) {
      continue;
    }
    setenv(key.c_str(), value.c_str(), /*overwrite=*/0);
  }
}

} // namespace

int main() {
  try {
    std::cout << termcolor::cyan << termcolor::bold
              << "Starting Ollama Client Chat Example..." << termcolor::reset
              << std::endl;

    loadDotEnvFile(LLM_CLIENT_ENV_FILE);

    // Ollama는 로컬(또는 사내망) 서버에 붙는 방식이라 API 키가 필요 없다.
    // OLLAMA_BASE_URL/OLLAMA_MODEL을 지정하지 않으면 각각
    // http://localhost:11434, llama3 기본값을 사용한다.
    std::string base_url = llm_client::Config::getEnv("OLLAMA_BASE_URL");
    std::string model = llm_client::Config::getEnv("OLLAMA_MODEL");

    auto ollama_client =
        llm_client::LLMClientFactory::create("ollama", /*api_key=*/"", base_url);

    llm_client::RequestParams params;
    if (!model.empty()) {
      params.model = model;
    }

    std::cout << "\nOllama(" << (base_url.empty() ? "http://localhost:11434"
                                                   : base_url)
              << ")와 채팅을 시작합니다." << std::endl;
    std::cout << "종료하려면 'exit' 또는 'quit'를 입력하세요.\n" << std::endl;

    replxx::Replxx rx;
    rx.set_max_history_size(100);

    // 이전 턴의 user/assistant 메시지를 누적해 실제 멀티턴 대화가
    // 되도록 한다 (매 요청마다 이전 대화 전체를 함께 전송).
    std::vector<llm_client::Message> conversation;

    while (true) {
      char const *input = rx.input("You: ");
      if (input == nullptr) {
        // EOF (Ctrl+D) 처리
        std::cout << termcolor::yellow << "\n채팅을 종료합니다."
                  << termcolor::reset << std::endl;
        break;
      }

      std::string user_input{input};

      if (user_input == "exit" || user_input == "quit") {
        std::cout << termcolor::yellow << "채팅을 종료합니다."
                  << termcolor::reset << std::endl;
        break;
      }

      if (user_input.empty()) {
        continue;
      }

      // 히스토리에 추가하여 위/아래 방향키로 탐색할 수 있도록 함
      rx.history_add(user_input);

      conversation.push_back({"user", user_input});

      try {
        auto response = ollama_client->chat(conversation, params);
        std::cout << termcolor::green << termcolor::bold
                  << "Ollama: " << termcolor::reset << termcolor::green
                  << response.content << termcolor::reset << std::endl;
        conversation.push_back({"assistant", response.content});
      } catch (const std::exception &e) {
        std::cerr << termcolor::red << "Ollama Error: " << e.what()
                  << termcolor::reset << std::endl;
        // 실패한 요청의 user 메시지는 이력에 남기지 않는다.
        conversation.pop_back();
      }
    }

  } catch (const std::exception &e) {
    std::cerr << termcolor::on_red << termcolor::white
              << "Global Error: " << e.what() << termcolor::reset << std::endl;
  }

  return 0;
}
