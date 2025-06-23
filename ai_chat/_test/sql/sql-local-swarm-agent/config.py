from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

class LLMProvider(Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"

@dataclass
class ModelConfig:
    provider: LLMProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000

# Model configurations
MODELS = {
    "gpt-3.5": ModelConfig(LLMProvider.OPENAI, "gpt-3.5-turbo"),
    "gpt-4": ModelConfig(LLMProvider.OPENAI, "gpt-4"),
    "gemini-pro": ModelConfig(LLMProvider.GOOGLE, "gemini-pro"),
    "gemini-pro-vision": ModelConfig(LLMProvider.GOOGLE, "gemini-pro-vision"),
    "gemini-ultra": ModelConfig(LLMProvider.GOOGLE, "gemini-ultra"),
    "qwen": ModelConfig(LLMProvider.OLLAMA, "qwen2.5:7b"),
}

# API configuration
API_CONFIG = {
    "openai": {
        "api_key": "",  # Set via environment variable OPENAI_API_KEY
        "organization": "",  # Optional, set via environment variable OPENAI_ORG_ID
    },
    "google": {
        "api_key": "",  # Set via environment variable GOOGLE_API_KEY
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
} 