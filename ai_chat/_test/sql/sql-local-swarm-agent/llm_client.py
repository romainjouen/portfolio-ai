import os
from typing import Optional, Dict, Any, List, Generator, Union
from dataclasses import dataclass
from openai import OpenAI
import google.generativeai as genai
from config import LLMProvider, ModelConfig, API_CONFIG, MODELS

@dataclass
class StreamingChunk:
    content: Optional[str] = None
    role: str = "assistant"
    finish_reason: Optional[str] = None
    provider: Optional[LLMProvider] = None

class ResponseAdapter:
    @staticmethod
    def adapt_streaming_response(response: Any, provider: LLMProvider) -> Generator[Dict[str, Any], None, None]:
        if provider == LLMProvider.GOOGLE:
            try:
                for chunk in response:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield {
                            "content": chunk.text,
                            "role": "assistant"
                        }
                # Final chunk to indicate completion
                yield {
                    "content": None,
                    "role": "assistant",
                    "finish_reason": "stop"
                }
            except Exception as e:
                print(f"Error adapting Gemini stream: {e}")
                if hasattr(response, 'text'):
                    content = response.text
                else:
                    content = str(response)
                yield {
                    "content": content,
                    "role": "assistant",
                    "finish_reason": "error"
                }
        else:
            # For OpenAI and Ollama
            try:
                if hasattr(response, '__iter__'):
                    for chunk in response:
                        # Handle OpenAI streaming response with dict
                        if isinstance(chunk, dict) and "choices" in chunk:
                            choice = chunk["choices"][0]
                            if "delta" in choice and choice["delta"].get("content") is not None:
                                yield {
                                    "content": choice["delta"]["content"],
                                    "role": "assistant",
                                    "finish_reason": choice.get("finish_reason")
                                }
                        # Handle Ollama streaming response
                        elif isinstance(chunk, dict):
                            if "content" in chunk:
                                yield {
                                    "content": chunk.get("content", ""),
                                    "role": chunk.get("role", "assistant"),
                                    "finish_reason": chunk.get("finish_reason")
                                }
                        else:
                            yield {
                                "content": str(chunk),
                                "role": "assistant"
                            }
                else:
                    # Handle non-iterable response
                    yield {
                        "content": str(response),
                        "role": "assistant",
                        "finish_reason": "stop"
                    }
            except Exception as e:
                print(f"Error adapting stream: {e}")
                yield {
                    "content": str(response),
                    "role": "assistant",
                    "finish_reason": "error"
                }

    @staticmethod
    def adapt_completion_response(response: Any, provider: LLMProvider) -> Dict[str, Any]:
        try:
            if provider == LLMProvider.GOOGLE:
                content = response.text if hasattr(response, 'text') else str(response)
                return {
                    "content": content,
                    "role": "assistant",
                    "finish_reason": "stop"
                }
            elif provider == LLMProvider.OLLAMA:
                if isinstance(response, dict):
                    return {
                        "content": response.get("content", str(response)),
                        "role": "assistant",
                        "finish_reason": response.get("finish_reason", "stop")
                    }
                else:
                    return {
                        "content": str(response),
                        "role": "assistant",
                        "finish_reason": "stop"
                    }
            else:  # OpenAI
                if hasattr(response, 'choices'):
                    message = response.choices[0].message
                    return {
                        "content": message.content,
                        "role": message.role,
                        "finish_reason": response.choices[0].finish_reason
                    }
                return {
                    "content": str(response),
                    "role": "assistant",
                    "finish_reason": "stop"
                }
        except Exception as e:
            print(f"Error adapting completion: {e}")
            return {
                "content": str(response),
                "role": "assistant",
                "finish_reason": "error"
            }

class ChatCompletions:
    def __init__(self, client_manager):
        self._client_manager = client_manager

    def create(self, model: str, messages: list, stream: bool = False, **kwargs):
        """Compatible with OpenAI's chat completion interface."""
        return self._client_manager.get_completion(model, messages, stream)

class Chat:
    def __init__(self, client_manager):
        self.completions = ChatCompletions(client_manager)

class SwarmCompatibleClient:
    """Wrapper class that provides the interface expected by Swarm."""
    def __init__(self, client_manager):
        self._client_manager = client_manager
        self.chat = Chat(client_manager)

class LLMClientManager:
    def __init__(self):
        self._clients: Dict[LLMProvider, Any] = {}
        self._initialize_clients()

    def _initialize_clients(self):
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                self._clients[LLMProvider.OPENAI] = OpenAI(
                    api_key=openai_api_key,
                    organization=os.getenv("OPENAI_ORG_ID")
                )
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")

        # Initialize Google client
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            try:
                genai.configure(api_key=google_api_key)
                self._clients[LLMProvider.GOOGLE] = genai
            except Exception as e:
                print(f"Failed to initialize Google client: {e}")

        # Initialize Ollama client
        try:
            ollama_config = API_CONFIG["ollama"]
            self._clients[LLMProvider.OLLAMA] = OpenAI(
                base_url=ollama_config["base_url"],
                api_key=ollama_config["api_key"]
            )
        except Exception as e:
            print(f"Failed to initialize Ollama client: {e}")

    def get_client(self, provider: LLMProvider) -> Optional[Any]:
        return self._clients.get(provider)

    def _get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration from model name."""
        for config in MODELS.values():
            if config.model_name == model_name:
                return config
        return None

    def _convert_messages_for_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Convert message format to work with Gemini.
        Gemini doesn't support chat history in the same way as OpenAI, so we'll concatenate messages."""
        formatted_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted_messages.append(f"System: {content}")
            elif role == "assistant":
                formatted_messages.append(f"Assistant: {content}")
            else:  # user
                formatted_messages.append(f"User: {content}")
        
        return "\n".join(formatted_messages)

    def get_completion(self, model_name: str, messages: list, stream: bool = False) -> Any:
        """Get completion using model name instead of config."""
        model_config = self._get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")

        client = self.get_client(model_config.provider)
        if not client:
            raise ValueError(f"No client initialized for provider {model_config.provider}")
        
        try:
            if model_config.provider == LLMProvider.OPENAI:
                response = client.chat.completions.create(
                    model=model_config.model_name,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    stream=stream
                )
                return response

            elif model_config.provider == LLMProvider.GOOGLE:
                model = client.GenerativeModel(model_config.model_name)
                # Convert messages to a format Gemini understands
                prompt = self._convert_messages_for_gemini(messages)
                response = model.generate_content(
                    prompt,
                    stream=stream,
                    generation_config=genai.types.GenerationConfig(
                        temperature=model_config.temperature,
                        max_output_tokens=model_config.max_tokens,
                    )
                )
                if stream:
                    return ResponseAdapter.adapt_streaming_response(response, LLMProvider.GOOGLE)
                else:
                    return ResponseAdapter.adapt_completion_response(response, LLMProvider.GOOGLE)

            elif model_config.provider == LLMProvider.OLLAMA:
                response = client.chat.completions.create(
                    model=model_config.model_name,
                    messages=messages,
                    temperature=model_config.temperature,
                    stream=stream
                )
                return response

        except Exception as e:
            print(f"Error getting completion from {model_config.provider}: {e}")
            raise

        return None 