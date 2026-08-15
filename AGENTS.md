# AGENTS.md - LLM_client 개발 및 TDD 가이드라인

이 문서는 `LLM_client` 프로젝트에서 작업하는 모든 AI 에이전트 및 개발자를 위한 **개발 원칙, TDD 워크플로우, 아키텍처 규칙**을 정의합니다. 본 프로젝트의 모든 코드 작성 및 수정 시 아래 가이드라인을 반드시 준수해야 합니다.

---

## 1. 프로젝트 개요 및 핵심 아키텍처

`LLM_client`는 OpenAI, Azure OpenAI, Ollama 및 커스텀 LLM 엔드포인트를 지원하는 고성능 C++17 클라이언트 라이브러리입니다.

### 📐 계층 구조 (Layer Architecture)
```
[ 언어별 바인딩 ] (Python / Go / Dart 등)
       │
[ C ABI 인터페이스 ] (include/llm_client/llm_client_c.h, src/llm_client_c.cpp)
       │
[ C++ Core 클라이언트 ] (OpenAIClient, AzureOpenAIClient, OllamaClient, ConversationSession)
       │ (의존성 주입 DI)
[ HTTP Transport 계층 ] (IHttpClient 추상 인터페이스)
    ├── [ CprHttpClient ]  (운영용: 실제 네트워크 cpr 통신)
    └── [ MockHttpClient ] (테스트용: 가짜 응답/SSE/에러 주입)
```

---

## 2. 🚨 TDD (Test-Driven Development) 필수 규칙

모든 신규 기능 구현, Provider 추가, 버그 수정 시 **TDD 원칙**을 엄격히 준수합니다.

### 2.1 절대 규칙 (Non-Negotiables)
1. **No Production Code Without Tests**: 실패하는 단위 테스트(`tests/test_*.cpp`)를 먼저 작성하지 않고 프로덕션 코드(`src/`)를 먼저 작성하지 마십시오.
2. **100% Network Isolation in Unit Tests**: 단위 테스트에서 실제 외부 LLM API(OpenAI, Ollama 등)를 호출해서는 안 됩니다. 반드시 [`MockHttpClient`](include/llm_client/mock_http_client.hpp)를 주입하여 0.1초 이내에 독립 실행되도록 작성하십시오.
3. **GoogleTest(GTest) 표준 준수**: `<cassert>`나 독립 `main()` 함수를 사용하지 마십시오. 모든 테스트는 `GTest::gtest`의 `TEST`, `TEST_F`, `EXPECT_*`, `ASSERT_*`를 사용해야 합니다.
4. **테스트 검증 의무화**: 코드 수정 후 반드시 `./manage.sh test`를 실행하여 기존 30개 이상의 단위 테스트와 신규 테스트가 모두 통과(`PASSED`)하는지 확인하십시오.

---

## 3. 신규 기능 및 Provider 추가 워크플로우

새로운 LLM Provider(예: Anthropic Claude, Google Gemini)나 새로운 옵션을 추가할 때는 다음 5단계를 거칩니다.

### Step 1: 실패하는 단위 테스트 작성 (Red)
`tests/test_<feature>.cpp`를 생성하거나 기존 테스트 파일에 GTest 테스트 케이스를 추가합니다.

```cpp
#include "llm_client/my_new_client.hpp"
#include "llm_client/mock_http_client.hpp"
#include <gtest/gtest.h>

TEST(MyNewClientTest, ChatSuccess_ParsesResponse) {
    auto mock_http = std::make_shared<llm_client::MockHttpClient>();
    mock_http->setResponse(200, R"({
        "result": "Hello from Mock!",
        "usage": {"total": 42}
    })");

    llm_client::MyNewClient client("fake-api-key", mock_http);
    auto res = client.generate("Ping");

    EXPECT_EQ(res.content, "Hello from Mock!");
    EXPECT_EQ(res.total_tokens, 42);
    EXPECT_EQ(mock_http->getLastUrl(), "https://api.example.com/v1/chat");
}
```

### Step 2: 의존성 주입(DI) 지원 클래스 선언 (Green)
헤더 파일(`include/llm_client/my_new_client.hpp`)에 `std::shared_ptr<IHttpClient> http_client = nullptr` 생성자를 지원합니다.

```cpp
#pragma once
#include "llm_client/http_client_interface.hpp"
#include "llm_client/llm_client_interface.hpp"
#include <memory>

namespace llm_client {

class MyNewClient : public LLMClientInterface {
public:
    explicit MyNewClient(const std::string &api_key,
                         std::shared_ptr<IHttpClient> http_client = nullptr);
    // ...
private:
    std::shared_ptr<IHttpClient> http_client_;
};

} // namespace llm_client
```

### Step 3: 최소한의 구현 작성 (Green)
`src/my_new_client.cpp`에서 `http_client ? http_client : std::make_shared<CprHttpClient>()` 형태로 초기화하고 `detail::http_post_json(*http_client_, ...)` 유틸 함수를 활용합니다.

### Step 4: CMake 등록 및 테스트 실행 (Verify & Refactor)
- `CMakeLists.txt`의 `llm_client` 및 `unit_tests` 소스 목록에 추가.
- `./manage.sh test`를 실행하여 전체 테스트 통과 확인.

### Step 5: C ABI 바인딩 확장 시 메모리 안전성 검증
`llm_client_c.cpp`에 함수를 추가한 경우, `tests/test_c_abi.cpp`에 C 구조체 메모리 해제(`llm_response_free`) 및 예외 매핑 테스트를 필수로 추가합니다.

---

## 4. MockHttpClient 활용 가이드

단위 테스트 작성 시 [`MockHttpClient`](include/llm_client/mock_http_client.hpp)에서 제공하는 기능:

| 기능 | 메서드 | 용도 |
| :--- | :--- | :--- |
| **일반 응답 설정** | `mock_http->setResponse(200, json_str)` | 단일 HTTP 응답 목킹 |
| **순차적 응답 설정** | `mock_http->queueResponse(HttpResponse{...})` | 호출 순서대로 다른 응답 반환 |
| **SSE 스트림 설정** | `mock_http->setStreamLines({"data: ...", ...})` | 실시간 스트리밍 청크 주입 |
| **네트워크 에러 시뮬레이션**| `mock_http->setError(7, "Connection refused")` | `NetworkException` 발생 검증 |
| **요청 본문 검증** | `mock_http->getLastJsonBody()` | 전송된 JSON 페이로드 필드 검증 |
| **요청 헤더 검증** | `mock_http->getLastHeaders()` | Authorization, Content-Type 검증 |
| **요청 URL 검증** | `mock_http->getLastUrl()` | 엔드포인트 URL 및 쿼리 파라미터 검증 |

---

## 5. 자주 사용하는 빌드 및 테스트 명령어

```bash
# 1. 전체 빌드 (CMake + Ninja)
./manage.sh build

# 2. 단위 테스트 전체 실행 (GTest)
./manage.sh test

# 3. 빌드 아티팩트 정리
./manage.sh clean

# 4. vcpkg 종속성 수동 동기화
./manage.sh deps

# 5. CTest 직접 실행
cd build && ctest --output-on-failure
```

---

## 6. 코딩 스타일 및 주의사항

- **C++ 표준**: C++17 표준 준수.
- **포인터 관리**: raw pointer 수동 메모리 할당(`new`) 금지 (`std::unique_ptr`, `std::shared_ptr` 사용).
- **C ABI (`src/llm_client_c.cpp`)**:
  - 문자열 반환 시 `c_strdup`을 사용하고, 호출자가 `llm_response_free`를 호출하여 해제할 수 있도록 일관성 유지.
  - 모든 C++ 예외는 `safe_invoke` 템플릿을 통해 적절한 `status_code`(1: Network, 2: API, 3: Parse, 4: Config, 5: General)로 변환할 것.
- **로깅**: `LLM_LOG_DEBUG`, `LLM_LOG_INFO`, `LLM_LOG_WARN`, `LLM_LOG_ERROR` 매크로 사용 ([include/llm_client/logger.hpp](include/llm_client/logger.hpp)).
