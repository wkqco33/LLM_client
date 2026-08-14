# C++ LLM Client Library

[![CI](https://github.com/wkqco33/LLM_client/actions/workflows/ci.yml/badge.svg)](https://github.com/wkqco33/LLM_client/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg)]()

다양한 LLM 프로바이더(OpenAI, Azure OpenAI, Ollama, Custom API)를 C++ 환경에서 일관되게 사용할 수 있도록 도와주는 경량 클라이언트 라이브러리입니다. 특정 프레임워크에 종속되지 않아 어떤 C++ 프로젝트에도 쉽게 통합할 수 있도록 설계되었으며, ROS2 패키지 역시 통합 가능한 예시 중 하나입니다.

## 특징 (Features)

- 단일 팩토리 인터페이스(`LLMClientFactory`)를 통한 다중 LLM 프로바이더 연동 지원
- `cpr`(HTTP Client)과 `nlohmann-json` 라이브러리를 사용한 모던 C++ 구현 (C++17), `vcpkg`로 의존성 관리
- 동기, SSE 기반 실시간 스트리밍 및 비동기 Future 기반 요청(`chatAsync`, `chatStreamAsync`, `generateAsync`) 지원
- `ContentBlock` 멀티모달(텍스트, 이미지 URL, Base64) 및 Structured Output(`response_format`, `json_schema`) 지원
- `cpr::Session` 기반 TCP/TLS 커넥션 재사용 및 Keep-Alive 네트워크 Latency 최적화
- 데이터 중심의 대화 맥락/슬라이딩 윈도우 관리자(`ConversationSession`) 지원
- HTTP 타임아웃(`timeout_ms`) 및 네트워크 오류/429/5xx 자동 재시도(`max_retries`, 지수 백오프 + `Retry-After` 헤더 반영) 내장
- Ollama(think) 등 reasoning 모델의 사고 과정을 `ResponseData.reasoning_content`로 수신 가능 (`RequestParams.thinking`)
- C ABI 래퍼(`llm_client_c.h`) 및 공식 Python 래퍼(`bindings/python/llm_client.py`) 제공으로 타 언어(Python, Go, Dart)에서도 쉽게 바인딩 가능
- RAG 등에 활용 가능한 임베딩(Embedding) API(`embed`, `embedAsync`) 지원 (OpenAI/Azure OpenAI/Ollama/Custom, C++/C ABI/Python 세 계층 모두 지원)
- 특정 프레임워크에 종속되지 않는 독립적인 의존성 그래프로 ROS2 등 어떤 C++ 프로젝트에도 통합 가능

## 프로바이더 지원 (Providers)

| 팩토리 키 | 설명 | 기본 모델 예시 | 임베딩(`embed`) 지원 |
|---|---|---|---|
| `openai` | `api.openai.com/v1` 호환 | `gpt-4o` | O (`text-embedding-3-small`) |
| `azure` | Microsoft Azure OpenAI Service (`base_url` 필수) | `gpt-4.1` (배포명 기준, reasoning 계열은 자동으로 `max_completion_tokens` 사용) | O (임베딩 모델 배포명 지정) |
| `ollama` | 로컬/원격 Ollama 서버 (`http://localhost:11434` 기본값) | `llama3` | O (`nomic-embed-text`, 별도 `ollama pull` 필요) |
| `custom` | OpenAI 호환 서버(vLLM 등), `base_url` 필수 | 사용자 지정 | 서버가 OpenAI 호환 `/embeddings`를 구현했다면 O |

각 프로바이더의 기본 모델은 `params.model`을 지정하지 않았을 때만 사용되는 기본값이며, 언제든 `RequestParams::model`(임베딩은 `EmbeddingParams::model`)로 원하는 모델/배포명을 지정할 수 있습니다.

---

## 빌드 및 설치 (Build Instructions)

본 라이브러리는 `vcpkg`를 패키지 매니저로 사용합니다. 사전에 vcpkg가 설치되어 있어야 합니다 (`VCPKG_ROOT` 환경 변수 또는 `~/Tools/vcpkg`, macOS Homebrew 환경을 자동 인식합니다).

제공되는 관리 스크립트(`manage.sh`)를 사용하면 쉽게 빌드할 수 있습니다:

```bash
# 권한 부여 (최초 1회)
chmod +x manage.sh

# 빌드 및 의존성 자동 설치 진행 (vcpkg manifest 모드로 cpr/nlohmann-json/spdlog 등 자동 설치)
./manage.sh build

# 빌드된 예제 실행해보기
./manage.sh run

# 프로젝트 초기화 (build 폴더 삭제)
./manage.sh clean
```

`manage.sh build`는 예제(`example_chat`, 기본적으로 Ollama에 연결하는 채팅 데모)도 함께 빌드합니다 (`replxx`, `termcolor`가 추가로 필요합니다). CMake를 직접 호출한다면 `-DLLM_CLIENT_BUILD_EXAMPLES=ON` 옵션을 전달하세요.

### CMakePresets.json으로 빌드하기 (대안)

`manage.sh` 대신 CMake 3.23+에 내장된 [Presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) 기능을 써도 됩니다. VSCode(CMake Tools)·CLion·`cmake` CLI가 모두 `CMakePresets.json`을 네이티브로 인식하므로 별도 스크립트 없이 프리셋을 선택/실행할 수 있습니다.

**사전 준비**: `manage.sh`와 달리 vcpkg 경로를 자동 탐색하지 않으므로, `VCPKG_ROOT` 환경 변수를 직접 설정해야 합니다.

```bash
export VCPKG_ROOT="$HOME/Tools/vcpkg"   # vcpkg를 설치한 실제 경로로 지정
```

제공되는 프리셋 목록:

| 프리셋 | 빌드 타입 | `example_chat` |
|---|---|---|
| `default` | Debug | 포함 |
| `release` | Release | 포함 |
| `lib-only` | Release | 미포함 (라이브러리만, `replxx`/`termcolor` 불필요) |

```bash
# 사용 가능한 프리셋 목록 확인
cmake --list-presets=all

# configure + build를 한 번에 (workflow preset, CMake 3.25+)
cmake --workflow --preset default

# 또는 configure/build를 각각 실행
cmake --preset release
cmake --build --preset release

# 라이브러리만 (예제 의존성 없이) 빌드하고 싶다면
cmake --workflow --preset lib-only
```

각 프리셋은 `build/<프리셋 이름>/` 아래에 독립된 빌드 디렉터리를 사용하므로 여러 프리셋을 서로 침범 없이 병행해서 쓸 수 있습니다. 빌드된 `example_chat`은 `build/default/example_chat`(또는 `build/release/example_chat`)에 생성됩니다.

로컬 환경에서만 필요한 값(예: 팀과 공유하지 않는 개인 `VCPKG_ROOT` 경로 오버라이드 등)이 있다면, git에 커밋되지 않는 `CMakeUserPresets.json`을 만들어 `CMakePresets.json`의 프리셋을 `inherits`로 확장해 사용하세요.

## 빠른 시작 (Quick Start)

### 기본 사용 (단발 요청)

```cpp
#include "llm_client/llm_client_factory.hpp"
#include <iostream>

int main() {
    try {
        // OpenAI 예시
        auto client = llm_client::LLMClientFactory::create("openai", "YOUR_API_KEY");

        // Azure OpenAI 예시 (base_url 필수, api_version은 미지정 시 기본값 사용)
        // auto azure_client = llm_client::LLMClientFactory::create(
        //     "azure", "YOUR_API_KEY", "https://YOUR_RESOURCE.openai.azure.com");

        // Ollama 예시 (로컬 서버, API 키 불필요)
        // auto ollama_client = llm_client::LLMClientFactory::create("ollama", "");

        llm_client::RequestParams params;
        params.temperature = 0.7f;
        params.max_tokens = 1024;
        params.timeout_ms = 30000; // 요청 타임아웃 (ms)
        params.max_retries = 2;    // 네트워크 오류/429/5xx 시 자동 재시도 횟수

        auto response = client->generate("안녕하세요! 자기소개를 해주세요.", params);
        std::cout << "Response: " << response.content << std::endl;
        std::cout << "Tokens used: " << response.total_tokens << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
```

### 스트리밍 사용 (`chatStream` / `generateStream`)

```cpp
auto client = llm_client::LLMClientFactory::create("openai", "YOUR_API_KEY");

llm_client::RequestParams params;
params.max_tokens = 1024;

auto response = client->generateStream(
    "짧은 시 한 편을 써줘.",
    [](const std::string &chunk) {
        std::cout << chunk << std::flush; // 토큰 단위로 즉시 출력
    },
    params);

std::cout << "\n--- finish_reason: " << response.finish_reason << " ---\n";
```

### Reasoning 모델의 사고 과정(thinking) 수신

Ollama 등의 reasoning 지원 모델은 `RequestParams::thinking = true`로 요청 시 사고 과정 텍스트를 `ResponseData::reasoning_content`로 함께 돌려줍니다 (최종 답변은 기존과 동일하게 `content`에 담깁니다).

```cpp
llm_client::RequestParams params;
params.thinking = true;

auto response = client->generate("12345 * 6789는?", params);
std::cout << "생각 과정: " << response.reasoning_content << std::endl;
std::cout << "최종 답변: " << response.content << std::endl;
```

> OpenAI/Azure OpenAI의 Chat Completions API는 reasoning 모델(o1/o3 등)의 사고 과정을 응답으로 노출하지 않으므로, 두 프로바이더에서는 `reasoning_content`가 채워지지 않습니다.

### 임베딩(Embedding) 사용

RAG(검색 증강 생성) 등에서 텍스트를 벡터로 변환할 때는 `embed()`/`embedAsync()`를 사용합니다. 벡터 저장/검색 자체는 이 라이브러리의 범위가 아니며, Qdrant 등 원하는 벡터 DB 클라이언트에 결과를 그대로 넘기면 됩니다.

```cpp
auto client = llm_client::LLMClientFactory::create("openai", "YOUR_API_KEY");

llm_client::EmbeddingParams params;
// params.model = "text-embedding-3-large"; // 미지정 시 프로바이더별 기본 임베딩 모델 사용

auto res = client->embed({"첫 번째 문장", "두 번째 문장"}, params);
for (const auto &vec : res.embeddings) {
    std::cout << "차원 수: " << vec.size() << std::endl;
}
```

`embedAsync()`는 `chatAsync()`와 동일하게 `std::future<EmbeddingResponse>`를 반환합니다.

C API에서는 `llm_client_embed()` / `llm_embedding_response_free()`를, Python에서는 `LLMClientPython.embed(inputs, model="")`를 동일한 방식으로 사용할 수 있습니다.

### `RequestParams` / `ResponseData` 필드 요약

| `RequestParams` | 설명 |
|---|---|
| `model` | 모델명/배포명 (미지정 시 프로바이더별 기본값 사용) |
| `temperature`, `top_p` | 샘플링 파라미터 |
| `max_tokens` | 최대 생성 토큰 수 |
| `thinking` | reasoning 모델의 사고 과정 활성화 여부 (Ollama 등 지원) |
| `timeout_ms` | HTTP 요청 타임아웃 (기본 30000ms, 스트리밍은 60000ms) |
| `max_retries` | 실패 시 자동 재시도 횟수 (기본 0 = 재시도 없음) |

| `ResponseData` | 설명 |
|---|---|
| `content` | 최종 응답 텍스트 |
| `reasoning_content` | thinking/reasoning 모델의 사고 과정 텍스트 |
| `model`, `finish_reason` | 실제 사용된 모델, 종료 사유 |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | 토큰 사용량 |

| `EmbeddingParams` | 설명 |
|---|---|
| `model` | 임베딩 모델명/배포명 (미지정 시 프로바이더별 기본값 사용) |
| `timeout_ms`, `max_retries` | `RequestParams`와 동일한 의미의 HTTP 타임아웃/재시도 옵션 |

| `EmbeddingResponse` | 설명 |
|---|---|
| `embeddings` | 입력 문자열 순서와 1:1 대응하는 벡터 목록 (`std::vector<std::vector<float>>`) |
| `model` | 실제 사용된 임베딩 모델 |
| `prompt_tokens`, `total_tokens` | 토큰 사용량 (일부 프로바이더는 미제공 시 0) |

### 프레임워크 연동 예시 (ROS2 등)

이 라이브러리는 특정 프레임워크에 종속되지 않는 범용 C++ 라이브러리이며, 어떤 CMake 기반 C++ 프로젝트에도 그대로 통합할 수 있습니다. 예를 들어 ROS2(rclcpp) 프로젝트에 통합하고 싶다면 `examples/ros2_node_example.cpp` 파일에 포함된 기본 예제를 참고하세요. 노드 초기화 과정에서 `llm_client::Config::getEnv()`를 통해 시스템 환경 변수 기반으로 동적 키를 할당하고 파라미터 구조를 활용하면 됩니다.

### C API / 타 언어 바인딩

`include/llm_client/llm_client_c.h`에 C ABI 호환 함수(`llm_client_create`, `llm_client_chat`, `llm_client_chat_stream`, `llm_response_free` 등)가 정의되어 있습니다. Go(`cgo`), Dart(`dart:ffi`), Python(`ctypes`) 등에서 정적/동적 라이브러리로 링크하여 사용할 수 있습니다.
