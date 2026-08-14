import sys
import os
import ctypes

# Add bindings path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../bindings/python")))

from llm_client import LLMClientPython, ResponseData

def test_python_binding():
    so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/libllm_client.so"))
    if not os.path.exists(so_path):
        print(f"Skipping python binding test because {so_path} does not exist yet.")
        return

    print(f"Loading shared library: {so_path}")
    client = LLMClientPython(so_path, "ollama", "")
    print("Python LLMClient object instantiated successfully!")

if __name__ == "__main__":
    test_python_binding()
