import sys
import os
import ctypes

# Add bindings path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../bindings/python")))

from llm_client import LLMClientPython, ResponseData

def test_python_binding():
    build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build"))
    candidates = [
        os.path.join(build_dir, "libllm_client.dylib"),
        os.path.join(build_dir, "libllm_client.so"),
        os.path.join(build_dir, "llm_client.dll")
    ]
    so_path = None
    for cand in candidates:
        if os.path.exists(cand):
            so_path = cand
            break

    if not so_path:
        print(f"Skipping python binding test because library was not found in {candidates}.")
        return

    print(f"Loading shared library: {so_path}")
    client = LLMClientPython(so_path, "ollama", "")
    print("Python LLMClient object instantiated successfully!")

if __name__ == "__main__":
    test_python_binding()
