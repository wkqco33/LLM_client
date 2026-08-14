#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace llm_client {

// 스트리밍 조각(chunk) 수신용 콜백 타입 정의
using StreamCallback = std::function<void(const std::string &chunk)>;

// 멀티모달 컨텐츠 블록 타입
enum class ContentType {
    Text,
    ImageUrl,
    ImageBase64
};

struct ContentBlock {
    ContentType type = ContentType::Text;
    std::string text;
    std::string image_url;
    std::string base64_data;
    std::string media_type = "image/jpeg"; // 예: "image/png", "image/jpeg", "image/webp"

    static ContentBlock makeText(std::string text_content) {
        ContentBlock cb;
        cb.type = ContentType::Text;
        cb.text = std::move(text_content);
        return cb;
    }

    static ContentBlock makeImageUrl(std::string url) {
        ContentBlock cb;
        cb.type = ContentType::ImageUrl;
        cb.image_url = std::move(url);
        return cb;
    }

    static ContentBlock makeImageBase64(std::string data, std::string mime = "image/jpeg") {
        ContentBlock cb;
        cb.type = ContentType::ImageBase64;
        cb.base64_data = std::move(data);
        cb.media_type = std::move(mime);
        return cb;
    }
};

// LLM에 전달되는 하나의 메시지 역할(Role)과 내용(Content)
struct Message {
    std::string role;    // "system", "user", "assistant" 등
    std::string content; // 프롬프트 텍스트 (blocks가 비어있을 때 사용)
    std::vector<ContentBlock> blocks; // 멀티모달 컨텐츠 블록들

    Message() = default;
    Message(std::string r, std::string c) : role(std::move(r)), content(std::move(c)) {}
    Message(std::string r, std::vector<ContentBlock> b) : role(std::move(r)), blocks(std::move(b)) {}
};

// LLM 요청 시 사용되는 파라미터 구조체
struct RequestParams {
    std::string model;
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<int> max_tokens;
    std::optional<bool> thinking;          // SLM / Reasoning 모델의 생각 과정(thinking) 활성화 여부
    std::optional<int> timeout_ms;         // HTTP 요청 타임아웃 (밀리초, 기본 30000ms 등)
    std::optional<int> max_retries;        // 실패 시 자동 재시도 횟수 (기본 0 = 재시도 없음)
    std::optional<std::string> response_format; // "json_object" 또는 "text"
    std::optional<std::string> json_schema;    // Structured Output을 위한 JSON Schema 문자열
};

// LLM 응답을 담는 구조체
struct ResponseData {
    std::string content;          // LLM이 반환한 최종 텍스트
    std::string reasoning_content;// thinking/reasoning 모델의 사고 과정 텍스트
    std::string model;            // 실제 사용된 모델
    std::string finish_reason;    // "stop", "length" 등
    int prompt_tokens = 0;        // 사용된 프롬프트 토큰 수
    int completion_tokens = 0;    // 생성된 토큰 수
    int total_tokens = 0;         // 전체 토큰 수
};

// 임베딩 요청 시 사용되는 파라미터 구조체
struct EmbeddingParams {
    std::string model;
    std::optional<int> timeout_ms;  // HTTP 요청 타임아웃 (밀리초)
    std::optional<int> max_retries; // 실패 시 자동 재시도 횟수
};

// 임베딩 응답을 담는 구조체
struct EmbeddingResponse {
    std::vector<std::vector<float>> embeddings; // 입력 문자열 순서와 1:1 대응
    std::string model;            // 실제 사용된 모델
    int prompt_tokens = 0;        // 사용된 프롬프트 토큰 수
    int total_tokens = 0;         // 전체 토큰 수
};

} // namespace llm_client
