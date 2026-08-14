cmake_minimum_required(VERSION 3.14)
project(llm_client VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    # Apple 환경에서 curl 등의 C 소스 컴파일 시 unistd.h 등과 충돌을 막기 위한 전역 플래그
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_DARWIN_C_SOURCE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_DARWIN_C_SOURCE")
    add_definitions(-D_DARWIN_C_SOURCE)
endif()

# vcpkg를 이용한 의존성 탐색
find_package(cpr CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# 인터페이스 및 라이브러리 정의
add_library(llm_client 
    src/llm_client.cpp
    src/openai_client.cpp
    src/azure_openai_client.cpp
    src/ollama_client.cpp
    src/llm_client_c.cpp
)
target_include_directories(llm_client PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# C ABI 공유 라이브러리 (Python/Go/Dart 바인딩용)
add_library(llm_client_shared SHARED
    src/llm_client.cpp
    src/openai_client.cpp
    src/azure_openai_client.cpp
    src/ollama_client.cpp
    src/llm_client_c.cpp
)
target_include_directories(llm_client_shared PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
set_target_properties(llm_client_shared PROPERTIES OUTPUT_NAME "llm_client")

# 외부 라이브러리 링크
target_link_libraries(llm_client PUBLIC
    cpr::cpr
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)
target_link_libraries(llm_client_shared PUBLIC
    cpr::cpr
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

option(LLM_CLIENT_BUILD_EXAMPLES "Build llm_client examples" OFF)
option(LLM_CLIENT_BUILD_TESTS "Build llm_client tests" ON)

if(LLM_CLIENT_BUILD_EXAMPLES)
    find_package(replxx CONFIG REQUIRED)
    find_path(TERMCOLOR_INCLUDE_DIRS "termcolor/termcolor.hpp")

    add_executable(example_chat examples/main.cpp)
    target_link_libraries(example_chat PRIVATE llm_client replxx::replxx)
    target_include_directories(example_chat PRIVATE ${TERMCOLOR_INCLUDE_DIRS})

    target_compile_definitions(example_chat PRIVATE
        "LLM_CLIENT_ENV_FILE=\"${CMAKE_CURRENT_SOURCE_DIR}/.env\"")
endif()

if(LLM_CLIENT_BUILD_TESTS)
    enable_testing()

    add_executable(test_async tests/test_async.cpp)
    target_link_libraries(test_async PRIVATE llm_client)
    add_test(NAME test_async COMMAND test_async)

    add_executable(test_multimodal tests/test_multimodal.cpp)
    target_link_libraries(test_multimodal PRIVATE llm_client)
    add_test(NAME test_multimodal COMMAND test_multimodal)

    add_executable(test_session tests/test_session.cpp)
    target_link_libraries(test_session PRIVATE llm_client)
    add_test(NAME test_session COMMAND test_session)

    add_executable(test_embedding tests/test_embedding.cpp)
    target_link_libraries(test_embedding PRIVATE llm_client)
    add_test(NAME test_embedding COMMAND test_embedding)
endif()
