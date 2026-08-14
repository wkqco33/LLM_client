# 다른 프로젝트에서 `llm_client` 라이브러리 연동하기 가이드

이 문서에서는 빌드된 `llm_client` C++ 라이브러리를 다른 CMake 기반 프로젝트에서 사용하는 방법을 설명합니다.

---

## 1. 시스템 수준에 설치하여 연동 (추천)

가장 깔끔한 방법은 이 라이브러리를 내 시스템에 컴파일 및 설치해둔 뒤, 다른 프로젝트에서 `find_package` 명령어를 사용하여 불러오는 것입니다.

### 1-1. 라이브러리 빌드 및 설치

먼저 이 저장소(`LLM_client`) 루트에서 제공된 관리 스크립트를 사용하여 설치를 진행합니다:

```bash
# 1. 빌드 진행
./manage.sh build

# 2. 시스템에 라이브러리 및 헤더 설치
./manage.sh install
```

*(권한이 부족할 경우 `sudo ./manage.sh install` 명령어를 사용할 수 있습니다.)*

### 1-2. 대상 프로젝트의 `CMakeLists.txt` 구성

라이브러리를 사용할 외부 프로젝트의 CMake 파일에 `llm_client` 패키지를 검색하고 링크시킵니다.
`llm_client`는 내부적으로 `cpr`, `nlohmann_json`, `spdlog`를 사용하므로 이들도 함께 링크될 것입니다 (vcpkg 툴체인을 통한 의존성 주입 권장).

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_awesome_app)

set(CMAKE_CXX_STANDARD 17)

# 1. llm_client 검색 (설치된 라이브러리 탐색)
find_package(llm_client REQUIRED)

# (선택) 외부 의존성들도 vcpkg를 통해 찾게 지정
find_package(cpr CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_executable(my_app main.cpp)

# 2. 라이브러리 링크
target_link_libraries(my_app PRIVATE 
    llm_client::llm_client
    cpr::cpr
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)
```

### 1-3. 대상 프로젝트 빌드

대상 프로젝트 역시 동일하게 `vcpkg` 툴체인을 사용하여 빌드하면 모든 종속성이 정상적으로 풀립니다.

```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE="$HOME/Tools/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

---

## 2. CMake FetchContent / Submodule로 포함하기

만약 라이브러리를 시스템에 설치하지 않고, 해당 소스 트리를 내 프로젝트에 종속시키는 구조를 띄고 싶다면 `add_subdirectory` 나 `FetchContent` 를 활용합니다.

### 2-1. `CMakeLists.txt` 설정

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_awesome_app)

set(CMAKE_CXX_STANDARD 17)

# 종속성 (vcpkg를 통함)
find_package(cpr CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# 1. 서브디렉토리로 추가 (git submodule add 등을 진행했다고 가정시)
add_subdirectory(third_party/LLM_client)

add_executable(my_app main.cpp)

# 2. 타겟 링크
target_link_libraries(my_app PRIVATE llm_client)
```

---

## 3. 코드 내 사용법 (C++ 소스코드)

두 연동 방식 중 어떤 방식을 사용하든, 코드 상에서의 사용법은 동일합니다.

`LLMClientFactory::create(provider, api_key, base_url = "", api_version = "")`의 `provider`로는
`"openai"`, `"azure"`, `"ollama"`, `"custom"`을 지정할 수 있습니다
(`azure`/`custom`은 `base_url`이 필수입니다).

```cpp
#include "llm_client/llm_client_factory.hpp"
#include <iostream>

int main() {
    try {
        // 원하는 프로바이더를 선택하여 클라이언트 생성

        // 1. OpenAI 예제
        // auto client = llm_client::LLMClientFactory::create("openai", "sk-...");

        // 2. Azure OpenAI 예제 (base_url 필수)
        auto client = llm_client::LLMClientFactory::create(
            "azure",
            "YOUR_AZURE_API_KEY",
            "https://YOUR_RESOURCE_NAME.openai.azure.com"
        );

        // 3. Ollama 예제 (로컬 서버, API 키 불필요)
        // auto client = llm_client::LLMClientFactory::create("ollama", "");

        llm_client::RequestParams params;
        params.max_tokens = 512;
        params.timeout_ms = 30000; // HTTP 타임아웃 (ms)
        params.max_retries = 2;    // 네트워크 오류/429/5xx 자동 재시도 횟수

        // 프롬프트 요청 생성
        auto response = client->generate("안녕하세요. 넌 누구니?", params);

        std::cout << "Response:\n" << response.content << std::endl;
        std::cout << "Tokens used: " << response.total_tokens << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "LLM Client 오류 발생: " << e.what() << std::endl;
    }

    return 0;
}
```

### 3-1. 스트리밍 응답 받기

토큰이 생성되는 대로 실시간으로 받고 싶다면 `chatStream`/`generateStream`에 콜백을 전달합니다:

```cpp
auto response = client->generateStream(
    "긴 이야기를 하나 들려줘.",
    [](const std::string &chunk) {
        std::cout << chunk << std::flush;
    },
    params);
```

### 3-2. Reasoning 모델의 사고 과정(thinking) 받기

Ollama 등의 reasoning 지원 모델에서 `params.thinking = true`로 설정하면 `response.reasoning_content`에
사고 과정 텍스트가 채워집니다 (OpenAI/Azure Chat Completions API는 미지원).

```cpp
params.thinking = true;
auto response = client->generate("12345 * 6789는?", params);
std::cout << "생각 과정: " << response.reasoning_content << std::endl;
```

### 3-3. 임베딩(Embedding) 사용하기

RAG 파이프라인에서 텍스트를 벡터로 변환해야 한다면 `embed()`(또는 비동기 `embedAsync()`)를 사용합니다. 다른 프로젝트에서 연동할 때도 동일한 API로 동작합니다.

```cpp
llm_client::EmbeddingParams params;
// params.model = "text-embedding-3-large"; // 미지정 시 프로바이더별 기본 임베딩 모델 사용

auto res = client->embed({"문서 조각 1", "문서 조각 2"}, params);
for (const auto &vec : res.embeddings) {
    // 벡터를 Qdrant 등 원하는 벡터 DB 클라이언트에 그대로 전달
}
```

지원 프로바이더는 `openai`/`azure`/`ollama`/`custom`입니다 (자세한 내용은 README의 프로바이더 표 참고).

### 3-4. 예제 실행하기 (`examples/main.cpp`)

`example_chat`은 기본적으로 **Ollama**(로컬 서버, API 키 불필요)에 연결하는 채팅 예제입니다.
실행 시점(런타임)에 `OLLAMA_BASE_URL`, `OLLAMA_MODEL` 환경 변수를 읽으며, 둘 다 비워두면
`http://localhost:11434`, `llama3` 기본값을 사용합니다. 셸에서 직접 export 하거나, 프로젝트
루트에 `.env` 파일을 두면 실행 파일이 시작 시 자동으로 읽어 들여 프로세스 환경 변수로 로드합니다
(이미 설정된 환경 변수는 덮어쓰지 않습니다). 값은 컴파일 타임에 바이너리로 굽히지 않으므로 빌드
산출물을 공유해도 노출되지 않습니다.

```bash
# 사전 준비: Ollama가 로컬에서 실행 중이고 모델이 pull 되어 있어야 합니다.
#   ollama pull llama3

cp .env.example .env
# 필요 시 .env 파일을 열어 OLLAMA_BASE_URL/OLLAMA_MODEL 수정
./manage.sh build   # example_chat도 함께 빌드됩니다
./manage.sh run
```

다른 프로바이더(OpenAI/Azure 등)로 직접 채팅해보고 싶다면 `examples/main.cpp`의
`LLMClientFactory::create("ollama", ...)` 호출부만 원하는 provider/키로 교체하면 됩니다
(위 3번 코드 예제 참고).
