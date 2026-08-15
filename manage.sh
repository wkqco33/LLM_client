#!/bin/bash

# 에러 발생 시 스크립트 중단
set -e

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${PROJECT_DIR}/build"

# VCPKG_ROOT 경로 탐색 (macOS Homebrew 및 ~/Tools 환경 고려)
if [ -z "$VCPKG_ROOT" ]; then
    if [ -d "$HOME/Tools/vcpkg" ]; then
        export VCPKG_ROOT="$HOME/Tools/vcpkg"
    elif command -v brew &> /dev/null; then
        BREW_PREFIX=$(brew --prefix)
        if [ -d "${BREW_PREFIX}/share/vcpkg" ]; then
            export VCPKG_ROOT="${BREW_PREFIX}/share/vcpkg"
        fi
    fi
fi

if [ -n "$VCPKG_ROOT" ]; then
    VCPKG_TOOLCHAIN="-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
else
    echo "⚠️ VCPKG_ROOT 환경 변수를 찾을 수 없습니다. CMake가 기본 설정으로 진행됩니다."
    VCPKG_TOOLCHAIN=""
fi

# OS에 따른 컴파일러 설정
# vcpkg의 macOS 빌드는 기본적으로 Apple Clang(libc++)을 사용합니다.
# Homebrew GCC(g++) 사용 시 ABI 충돌(std::__1 vs std::__cxx11)이 발생하므로 macOS에서는 강제 설정합니다.
if [ "$(uname)" = "Darwin" ]; then
    export CC=clang
    export CXX=clang++
fi

function print_help() {
    echo "사용법: ./manage.sh [명령어]"
    echo ""
    echo "명령어:"
    echo "  build    - 빌드 및 의존성 설치 진행"
    echo "  test     - 단위 테스트 (GoogleTest / CTest) 실행"
    echo "  clean    - 빌드 결과물(build 폴더) 삭제"
    echo "  deps     - vcpkg를 통한 종속성 수동 설치"
    echo "  install  - 빌드된 결과물을 시스템에 설치"
    echo "  uninstall- 시스템에 설치된 결과물 삭제"
    echo "  run      - 예제 프로그램 실행"
    echo "  help     - 도움말 출력"
    echo ""
}

function clean() {
    echo "🧹 빌드 디렉토리 삭제 중..."
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        echo "✅ 완료: 프로젝트가 초기화되었습니다."
    else
        echo "✅ 정리할 빌드 디렉토리가 없습니다."
    fi
}

function build() {
    echo "🔨 빌드 시작..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # CMake Configure (run 서브커맨드로 실행할 example_chat도 함께 빌드)
    if command -v ninja &> /dev/null; then
        cmake .. $VCPKG_TOOLCHAIN -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLM_CLIENT_BUILD_EXAMPLES=ON -G Ninja
    else
        cmake .. $VCPKG_TOOLCHAIN -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLM_CLIENT_BUILD_EXAMPLES=ON
    fi
    
    # CMake Build
    cmake --build . --parallel
    
    # vscode/clangd를 위한 compile_commands.json 복사
    if [ -f compile_commands.json ]; then
        cp compile_commands.json "$PROJECT_DIR/" || true
    fi
    echo "✅ 빌드 완료"
}

function deps() {
    echo "📦 종속성 설치 확인 중..."
    if [ -n "$VCPKG_ROOT" ] && [ -x "$VCPKG_ROOT/vcpkg" ]; then
        "$VCPKG_ROOT/vcpkg" install
        echo "✅ 종속성 설치 완료"
    elif command -v vcpkg &> /dev/null; then
        vcpkg install
        echo "✅ 종속성 설치 완료"
    else
        echo "❌ vcpkg를 찾을 수 없습니다. 설치 또는 VCPKG_ROOT 설정을 확인하세요."
        exit 1
    fi
}

function install_target() {
    echo "📥 설치 진행 중..."
    if [ ! -d "$BUILD_DIR" ]; then
        echo "❌ 빌드 폴더가 없습니다. 먼저 빌드를 진행하세요."
        exit 1
    fi
    cd "$BUILD_DIR"
    cmake --install .
    echo "✅ 설치 완료"
}

function uninstall_target() {
    echo "🗑️ 설치 제거 진행 중..."
    if [ ! -f "$BUILD_DIR/install_manifest.txt" ]; then
        echo "❌ 설치 매니페스트(install_manifest.txt)를 찾을 수 없습니다. (install 된 적이 없거나 빌드 폴더가 삭제됨)"
        exit 1
    fi
    # 매니페스트 파일에 명시된 파일들을 일괄 삭제합니다.
    # 권한 문제 시 sudo가 필요할 수 있으므로, 에러 발생 시 안내 메시지를 띄우는 것이 좋습니다.
    if xargs rm -f < "$BUILD_DIR/install_manifest.txt" 2>/dev/null; then
        echo "✅ 설치 제거 완료"
    else
        echo "⚠️ 일부 파일을 삭제하지 못했습니다. 관리자 권한(sudo ./manage.sh uninstall)이 필요할 수 있습니다."
    fi
}

function run() {
    echo "🚀 예제 실행 중..."
    if [ -f "$BUILD_DIR/example_chat" ]; then
        "$BUILD_DIR/example_chat"
    else
        echo "❌ 실행 파일(example_chat)을 찾을 수 없습니다. 먼저 빌드해주세요."
        exit 1
    fi
}

function run_tests() {
    echo "🧪 단위 테스트 실행 중..."
    if [ ! -d "$BUILD_DIR" ]; then
        echo "⚠️ 빌드 폴더가 없습니다. 먼저 빌드를 진행합니다."
        build
    fi
    cd "$BUILD_DIR"
    if [ -f "unit_tests" ]; then
        ./unit_tests --gtest_color=yes
    else
        ctest --output-on-failure
    fi
    echo "✅ 테스트 완료"
}

if [ $# -eq 0 ]; then
    print_help
    exit 0
fi

case "$1" in
    build) build ;;
    test) run_tests ;;
    clean) clean ;;
    deps) deps ;;
    install) install_target ;;
    uninstall) uninstall_target ;;
    run) run ;;
    help|--help|-h) print_help ;;
    *)
        echo "❌ 지원하지 않는 명령어: $1"
        print_help
        exit 1
        ;;
esac
