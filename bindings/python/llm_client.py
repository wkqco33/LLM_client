"""
C ABI wrapper for C++ LLM Client using ctypes
"""
import ctypes
import os
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass

class LLMClientHandle(ctypes.c_void_p):
    pass

class LLMMessageStruct(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("content", ctypes.c_char_p)
    ]

class LLMRequestParamsStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p),
        ("temperature", ctypes.c_float),
        ("has_temperature", ctypes.c_bool),
        ("top_p", ctypes.c_float),
        ("has_top_p", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
        ("has_max_tokens", ctypes.c_bool),
        ("thinking", ctypes.c_bool),
        ("has_thinking", ctypes.c_bool),
        ("timeout_ms", ctypes.c_int),
        ("max_retries", ctypes.c_int)
    ]

class LLMResponseDataStruct(ctypes.Structure):
    _fields_ = [
        ("content", ctypes.c_char_p),
        ("reasoning_content", ctypes.c_char_p),
        ("model", ctypes.c_char_p),
        ("finish_reason", ctypes.c_char_p),
        ("prompt_tokens", ctypes.c_int),
        ("completion_tokens", ctypes.c_int),
        ("total_tokens", ctypes.c_int),
        ("status_code", ctypes.c_int),
        ("error_message", ctypes.c_char_p)
    ]

class LLMEmbeddingParamsStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p),
        ("timeout_ms", ctypes.c_int),
        ("max_retries", ctypes.c_int)
    ]

class LLMEmbeddingResponseStruct(ctypes.Structure):
    _fields_ = [
        ("embeddings", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        ("embedding_dims", ctypes.POINTER(ctypes.c_size_t)),
        ("count", ctypes.c_size_t),
        ("model", ctypes.c_char_p),
        ("prompt_tokens", ctypes.c_int),
        ("total_tokens", ctypes.c_int),
        ("status_code", ctypes.c_int),
        ("error_message", ctypes.c_char_p)
    ]

STREAM_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)

@dataclass
class ResponseData:
    content: str
    reasoning_content: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    status_code: int
    error_message: str

@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    model: str
    prompt_tokens: int
    total_tokens: int
    status_code: int
    error_message: str

class LLMClientPython:
    def __init__(self, lib_path: str, provider: str, api_key: str, base_url: str = "", api_version: str = ""):
        self.lib = ctypes.CDLL(lib_path)
        
        self.lib.llm_client_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.llm_client_create.restype = LLMClientHandle
        
        self.lib.llm_client_destroy.argtypes = [LLMClientHandle]
        self.lib.llm_client_destroy.restype = None
        
        self.lib.llm_client_generate.argtypes = [LLMClientHandle, ctypes.c_char_p, ctypes.POINTER(LLMRequestParamsStruct)]
        self.lib.llm_client_generate.restype = LLMResponseDataStruct

        self.lib.llm_response_free.argtypes = [ctypes.POINTER(LLMResponseDataStruct)]
        self.lib.llm_response_free.restype = None

        self.lib.llm_client_embed.argtypes = [
            LLMClientHandle,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_size_t,
            ctypes.POINTER(LLMEmbeddingParamsStruct)
        ]
        self.lib.llm_client_embed.restype = LLMEmbeddingResponseStruct

        self.lib.llm_embedding_response_free.argtypes = [ctypes.POINTER(LLMEmbeddingResponseStruct)]
        self.lib.llm_embedding_response_free.restype = None

        self.handle = self.lib.llm_client_create(
            provider.encode('utf-8'),
            api_key.encode('utf-8'),
            base_url.encode('utf-8') if base_url else None,
            api_version.encode('utf-8') if api_version else None
        )
        if not self.handle:
            raise RuntimeError(f"Failed to create LLM client for provider '{provider}'")

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.lib.llm_client_destroy(self.handle)
            self.handle = None

    def generate(self, prompt: str, model: str = "", temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> ResponseData:
        params = LLMRequestParamsStruct()
        params.model = model.encode('utf-8') if model else None
        if temperature is not None:
            params.temperature = temperature
            params.has_temperature = True
        if max_tokens is not None:
            params.max_tokens = max_tokens
            params.has_max_tokens = True

        raw_res = self.lib.llm_client_generate(self.handle, prompt.encode('utf-8'), ctypes.byref(params))
        
        res = ResponseData(
            content=raw_res.content.decode('utf-8') if raw_res.content else "",
            reasoning_content=raw_res.reasoning_content.decode('utf-8') if raw_res.reasoning_content else "",
            model=raw_res.model.decode('utf-8') if raw_res.model else "",
            finish_reason=raw_res.finish_reason.decode('utf-8') if raw_res.finish_reason else "",
            prompt_tokens=raw_res.prompt_tokens,
            completion_tokens=raw_res.completion_tokens,
            total_tokens=raw_res.total_tokens,
            status_code=raw_res.status_code,
            error_message=raw_res.error_message.decode('utf-8') if raw_res.error_message else ""
        )
        
        self.lib.llm_response_free(ctypes.byref(raw_res))
        return res

    def embed(self, inputs: List[str], model: str = "") -> EmbeddingResponse:
        params = LLMEmbeddingParamsStruct()
        params.model = model.encode('utf-8') if model else None

        encoded_inputs = [s.encode('utf-8') for s in inputs]
        c_inputs = (ctypes.c_char_p * len(encoded_inputs))(*encoded_inputs)

        raw_res = self.lib.llm_client_embed(self.handle, c_inputs, len(inputs), ctypes.byref(params))

        embeddings: List[List[float]] = []
        for i in range(raw_res.count):
            dim = raw_res.embedding_dims[i]
            row_ptr = raw_res.embeddings[i]
            embeddings.append([row_ptr[j] for j in range(dim)] if row_ptr else [])

        res = EmbeddingResponse(
            embeddings=embeddings,
            model=raw_res.model.decode('utf-8') if raw_res.model else "",
            prompt_tokens=raw_res.prompt_tokens,
            total_tokens=raw_res.total_tokens,
            status_code=raw_res.status_code,
            error_message=raw_res.error_message.decode('utf-8') if raw_res.error_message else ""
        )

        self.lib.llm_embedding_response_free(ctypes.byref(raw_res))
        return res
