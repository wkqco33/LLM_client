# Change Log

이 프로젝트의 변경 사항을 기록합니다.

## [0.2.0] - TDD 인프라 및 로컬 ONNX 모델 추론 지원
 
### 신규 추가 (New Features)
- **로컬 ONNX 모델 추론 클라이언트 (`OnnxClient`)**
  - 로컬 `.onnx` 파일 기반 텍스트 생성(`chat`, `chatStream`) 및 임베딩(`embed`) 지원
  - HuggingFace `tokenizer.json` 및 `vocab.json`을 지원하는 자체 토크나이저(`SimpleTokenizer`) 구현
  - Greedy / Temperature / Top-P (Nucleus) 샘플링 지원
  - 임베딩 생성 시 `last_hidden_state` 기반 Attention-weighted Mean Pooling 지원
  - `LLMClientFactory` 및 C ABI에 `"onnx"` 프로바이더 연동

- **TDD (Test-Driven Development) 아키텍처 및 GoogleTest 도입**
  - HTTP 통신 추상화 인터페이스(`IHttpClient`) 및 의존성 주입(DI) 계층 구축
  - 운영용 `CprHttpClient` 및 테스트용 `MockHttpClient` 분리
  - ONNX 런타임 엔진 추상화(`IOnnxEngine`) 및 `MockOnnxEngine` 제공
  - 9개 테스트 스위트, 35개 단위 테스트로 구성된 GoogleTest 기반 단위 테스트 프레임워크 구축 (`./manage.sh test`)
  - AI 에이전트 및 개발자를 위한 [AGENTS.md](AGENTS.md) TDD 개발 가이드라인 추가

---

## [0.1.0] - 초기 공개 (Initial Release)

### 신규 추가 (범용 C++ LLM 클라이언트 라이브러리 첫 공개)

- **단일 팩토리 인터페이스 (`LLMClientFactory`)**
  - `openai`, `azure`, `ollama`, `custom`(OpenAI 호환 서버) 4개 프로바이더를 하나의 `create()`로 통일 지원
  - 특정 프레임워크에 종속되지 않는 독립적인 C++17 라이브러리 (ROS2 등 어떤 CMake 프로젝트에도 통합 가능)

- **동기 · SSE 스트리밍 · 비동기 API**
  - 동기: `chat`, `generate`
  - 스트리밍(SSE 실시간 토큰): `chatStream`, `generateStream`
  - 비동기(`std::future`): `chatAsync`, `chatStreamAsync`, `generateAsync`, `generateStreamAsync`
  - HTTP 커넥션 재사용(`cpr::Session`) 및 Keep-Alive 기반 네트워크 지연 최적화

- **멀티모달 및 Structured Output**
  - `ContentBlock`(텍스트 / 이미지 URL / Base64) 기반 멀티모달 프롬프트
  - `RequestParams::response_format` / `json_schema` 로 JSON Structured Output 지원
  - Ollama 등 reasoning 모델의 사고 과정을 `ResponseData::reasoning_content`로 수신 (`RequestParams::thinking`)

- **임베딩(Embedding) API (RAG 대응)**
  - `embed` / `embedAsync` 지원 (`EmbeddingParams`, `EmbeddingResponse`)
  - OpenAI, Azure OpenAI, Ollama, Custom 프로바이더 지원 (OpenAI 호환 `/embeddings` 구현 시)
  - 벡터 저장/검색(Qdrant 등)은 라이브러리 범위 밖이며, `embed()` 결과를 그대로 원하는 벡터 DB 클라이언트에 전달

- **대화 세션 컨텍스트 관리 (`ConversationSession`)**
  - 시스템 프롬프트 보존 및 슬라이딩 윈도우 기반 대화 이력 자동 트리밍

- **타임아웃 & 자동 재시도**
  - `RequestParams::timeout_ms` (HTTP 요청 타임아웃)
  - `RequestParams::max_retries` (네트워크 오류 / 429 / 5xx 시 지수 백오프 + `Retry-After` 헤더 반영 자동 재시도)

- **에러/로깅 일원화 (`http_util.hpp`)**
  - 공통 HTTP 헬퍼(`http_post_json`, `http_post_stream`)로 프로바이더 간 중복 제거
  - 네트워크 오류(`NetworkException`), JSON 파싱 실패(`ParseException`), 설정 오류(`ConfigurationException`) 구분
  - 응답 JSON 안전 접근(Safe Field Check)으로 키/배열 부재 시 크래시 방지
  - `spdlog` 기반 로깅

- **C ABI & 타 언어 바인딩**
  - `include/llm_client/llm_client_c.h` C ABI 래퍼 (`llm_client_create`, `llm_client_chat`, `llm_client_chat_stream`, `llm_client_embed`, `llm_response_free` 등)
  - 공식 Python 래퍼 `bindings/python/llm_client.py` (ctypes 기반, `generate` / `embed` 지원)
  - Go(`cgo`), Dart(`dart:ffi`), Python(`ctypes`) 등에서 정적/동적 라이브러리로 링크 가능

### 의존성 및 빌드
- C++17, `cpr`, `nlohmann-json`, `spdlog` — `vcpkg`(manifest 모드)로 의존성 관리
- `CMakePresets.json` 프리셋 제공 (`default`/`release`/`lib-only`)
- `manage.sh` 관리 스크립트 (`build` / `run` / `clean` / `deps` / `install` / `uninstall`)
- `libllm_client.so` 공유 라이브러리 타겟 포함
