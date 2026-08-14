#ifndef LLM_CLIENT_C_H
#define LLM_CLIENT_C_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(LLM_CLIENT_BUILD_DLL)
    #define LLM_C_API __declspec(dllexport)
  #else
    #define LLM_C_API __declspec(dllimport)
  #endif
#else
  #define LLM_C_API __attribute__((visibility("default")))
#endif

/**
 * @brief LLM 클라이언트 인스턴스 핸들 타입 (opaque pointer)
 */
typedef void *llm_client_h;

/**
 * @brief C 호환 메시지 구조체
 */
typedef struct {
  const char *role;    // "user", "assistant", "system"
  const char *content; // 메시지 본문
} llm_message_t;

/**
 * @brief C 호환 요청 파라미터 구조체
 */
typedef struct {
  const char *model;
  float temperature;
  bool has_temperature;
  float top_p;
  bool has_top_p;
  int max_tokens;
  bool has_max_tokens;
  bool thinking;
  bool has_thinking;
  int timeout_ms;  // 0 지정 시 기본값 사용
  int max_retries; // 0 지정 시 재시도 안 함
} llm_request_params_t;

/**
 * @brief C 호환 응답 데이터 구조체
 */
typedef struct {
  char *content;           // 힙 할당 문자열 (llm_response_free로 해제 필요)
  char *reasoning_content; // 힙 할당 사고 과정 문자열
  char *model;             // 사용된 모델명
  char *finish_reason;     // 종료 사유
  int prompt_tokens;       // 프롬프트 토큰 수
  int completion_tokens;   // 완료 토큰 수
  int total_tokens;        // 전체 토큰 수
  int status_code;         // 0: 성공, 1: NetworkError, 2: APIError, 3: ParseError, 4: ConfigError
  char *error_message;     // 에러 발생 시 상세 메시지
} llm_response_data_t;

/**
 * @brief C 호환 스트리밍 콜백 함수 포인터
 */
typedef void (*llm_stream_callback_t)(const char *chunk, void *user_data);

/**
 * @brief C 호환 임베딩 요청 파라미터 구조체
 */
typedef struct {
  const char *model;
  int timeout_ms;  // 0 지정 시 기본값 사용
  int max_retries; // 0 지정 시 재시도 안 함
} llm_embedding_params_t;

/**
 * @brief C 호환 임베딩 응답 데이터 구조체
 */
typedef struct {
  float **embeddings;      // 힙 할당된 count개의 float* 배열 (llm_embedding_response_free로 해제 필요)
  size_t *embedding_dims;  // 각 embeddings[i]의 차원 수 (길이: count)
  size_t count;            // 임베딩 개수 (입력 문자열 개수와 동일)
  char *model;              // 사용된 모델명
  int prompt_tokens;       // 사용된 프롬프트 토큰 수
  int total_tokens;        // 전체 토큰 수
  int status_code;         // 0: 성공, 1: NetworkError, 2: APIError, 3: ParseError, 4: ConfigError
  char *error_message;     // 에러 발생 시 상세 메시지
} llm_embedding_response_t;

/**
 * @brief LLM 클라이언트 생성
 * @param provider 프로바이더 이름 ("openai", "azure", "ollama", "custom")
 * @param api_key API 키
 * @param base_url 기본 엔드포인트 URL (선택사항, NULL 설정 시 기본값)
 * @param api_version API 버전 (Azure 선택사항)
 * @return 생성된 클라이언트 핸들 (실패 시 NULL)
 */
LLM_C_API llm_client_h llm_client_create(const char *provider,
                                          const char *api_key,
                                          const char *base_url,
                                          const char *api_version);

/**
 * @brief LLM 클라이언트 해제
 */
LLM_C_API void llm_client_destroy(llm_client_h client);

/**
 * @brief 대화 요청 및 응답 생성 (동기)
 */
LLM_C_API llm_response_data_t
llm_client_chat(llm_client_h client, const llm_message_t *messages,
                size_t message_count, const llm_request_params_t *params);

/**
 * @brief 단일 프롬프트 생성 편의 함수 (동기)
 */
LLM_C_API llm_response_data_t
llm_client_generate(llm_client_h client, const char *prompt,
                    const llm_request_params_t *params);

/**
 * @brief 스트리밍 대화 요청 (실시간 콜백)
 */
LLM_C_API llm_response_data_t llm_client_chat_stream(
    llm_client_h client, const llm_message_t *messages, size_t message_count,
    llm_stream_callback_t callback, void *user_data,
    const llm_request_params_t *params);

/**
 * @brief 응답 구조체 내부의 힙 메모리 해제
 */
LLM_C_API void llm_response_free(llm_response_data_t *response);

/**
 * @brief 텍스트 배열을 임베딩 벡터로 변환 (RAG 등에서 사용, 동기)
 * @param inputs 임베딩할 문자열 배열
 * @param input_count inputs 배열의 길이
 */
LLM_C_API llm_embedding_response_t
llm_client_embed(llm_client_h client, const char **inputs, size_t input_count,
                  const llm_embedding_params_t *params);

/**
 * @brief 임베딩 응답 구조체 내부의 힙 메모리 해제
 */
LLM_C_API void llm_embedding_response_free(llm_embedding_response_t *response);

#ifdef __cplusplus
}
#endif

#endif // LLM_CLIENT_C_H
