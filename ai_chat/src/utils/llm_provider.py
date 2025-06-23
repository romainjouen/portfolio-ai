from typing import List, Dict, Optional, Any
import os
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from huggingface_hub import InferenceClient
import requests
import json
from ..config.config import Config
import groq
import logging

class LLMProvider:
    def __init__(self, selected_provider: str = None):
        # Load config
        config = Config()
        self.config = config
        self.model_families = config.model_families
        self.selected_provider = selected_provider

        # Initialize API keys from environment
        self.api_keys = {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'mistral': os.getenv('MISTRAL_API_KEY'),
            'huggingface': os.getenv('HUGGINGFACE_API_KEY'),
            'gemini': os.getenv('GEMINI_API_KEY'),
            'deepseek': os.getenv('DEEPSEEK_API_KEY'),
            'groq': os.getenv('GROQ_API_KEY')
        }

        # Initialize token and cost tracking
        self.input_tokens = 0
        self.output_tokens = 0
        self.cumulative_cost = 0.0

        # Initialize clients
        self.clients = {}
        if selected_provider:
            self._init_client(selected_provider)

    def _init_client(self, provider: str):
        """Initialize specific client based on provider"""
        try:
            # Get model configuration
            model_config = self._get_model_config(provider)
            family = provider.split('_')[0]

            if family == 'openai' and self.api_keys['openai']:
                self.clients['openai'] = OpenAI(api_key=self.api_keys['openai'])
            
            elif family == 'anthropic' and self.api_keys['anthropic']:
                self.clients['anthropic'] = Anthropic(api_key=self.api_keys['anthropic'])
            
            elif family == 'gemini' and self.api_keys['gemini']:
                genai.configure(api_key=self.api_keys['gemini'])
                self.clients['gemini'] = genai.GenerativeModel(model_config['name'])
            
            elif family == 'huggingface' and self.api_keys['huggingface']:
                self.clients['huggingface'] = InferenceClient(token=self.api_keys['huggingface'])
            
            elif family == 'deepseek' and self.api_keys['deepseek']:
                base_url = self.model_families['deepseek'].get('base_url', "https://api.deepseek.com/v1")
                self.clients['deepseek'] = OpenAI(
                    api_key=self.api_keys['deepseek'],
                    base_url=base_url
                )
                print("DeepSeek client initialized successfully")
            
            elif family == 'groq' and self.api_keys['groq']:
                self.clients['groq'] = groq.Groq(api_key=self.api_keys['groq'])
                print("Groq client initialized successfully")
            
            else:
                raise ValueError(f"Unsupported or unconfigured provider family: {family}")
                
        except Exception as e:
            print(f"Error initializing client for {provider}: {str(e)}")
            raise

    def _get_model_config(self, provider: str) -> Dict:
        """Get model configuration for a provider"""
        try:
            # Split provider into family and model
            parts = provider.split('_')
            if len(parts) != 2:
                raise ValueError(f"Invalid provider format: {provider}")
            
            family, model = parts
            
            # Get family config
            family_config = self.model_families.get(family)
            if not family_config:
                raise ValueError(f"No configuration found for family: {family}")
            
            # Get model config
            model_config = family_config.get('models', {}).get(model)
            if not model_config:
                raise ValueError(f"No configuration found for model: {model} in family {family}")
            
            # Include family-level configs if needed
            if 'base_url' in family_config:
                model_config['base_url'] = family_config['base_url']
            
            return model_config
            
        except Exception as e:
            print(f"Error getting model config for {provider}: {str(e)}")
            raise

    def _create_system_message(self, provider: str) -> Dict:
        """Create system message for a provider"""
        return {
            "role": "system",
            "content": self.config.prompts['chat']['system_message']
        }

    def _update_usage_stats(self, input_tokens: int, output_tokens: int, model_config: Dict) -> float:
        """Update token and cost tracking"""
        # Update token counts
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        
        # Calculate cost
        input_cost = (input_tokens * model_config['input_cost'] / 1000)
        output_cost = (output_tokens * model_config['output_cost'] / 1000)
        total_cost = input_cost + output_cost
        
        # Update cumulative cost
        self.cumulative_cost += total_cost
        
        return total_cost

    def _log_response(self, provider: str, input_tokens: int, output_tokens: int, cost: float, result: str):
        """Log response details"""
        print(f"\n=== {provider} API Response ===")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Total tokens: {input_tokens + output_tokens}")
        print(f"This call cost: ${cost:.5f}")
        print(f"Cumulative input tokens: {self.input_tokens}")
        print(f"Cumulative output tokens: {self.output_tokens}")
        print(f"Cumulative cost: ${self.cumulative_cost:.5f}")
        print(f"Response: {result[:100]}...")

    def process_query(self, provider: str, messages: List[Dict], temperature: float = 0.1) -> Optional[str]:
        """Process query with specified provider"""
        try:
            # Initialize client if not already initialized or if provider changed
            if provider != self.selected_provider:
                self.selected_provider = provider
                self._init_client(provider)

            model_config = self._get_model_config(provider)
            system_message = self._create_system_message(provider)
            
            # Get only the last message (latest user query)
            last_message = messages[-1] if messages else None
            if not last_message:
                raise ValueError("No message provided")
            
            # Use only system message and last user message for the API call
            current_messages = [system_message, last_message]
            
            # Get client for provider family
            family = provider.split('_')[0]
            client = self.clients.get(family)
            if not client:
                raise ValueError(f"No client initialized for provider family: {family}")
            
            # Get response using provider-specific method
            response = self._get_provider_response(provider, client, current_messages, temperature, model_config)
            
            return response
            
        except Exception as e:
            print(f"Error with {provider}: {str(e)}")
            return None

    def _get_provider_response(self, provider: str, client: Any, messages: List[Dict], 
                             temperature: float, model_config: Dict) -> str:
        """Get response from specific provider"""
        try:
            family = provider.split('_')[0]
            
            if family == 'anthropic':
                return self._anthropic_response(client, messages, temperature, model_config)
            elif family == 'openai':
                return self._openai_response(client, messages, temperature, model_config)
            elif family == 'gemini':
                return self._gemini_response(client, messages, temperature, model_config)
            elif family == 'mistral':
                return self._mistral_response(client, messages, temperature, model_config)
            elif family == 'deepseek':
                return self._deepseek_response(client, messages, temperature, model_config)
            elif family == 'groq':
                return self._groq_response(client, messages, temperature, model_config)
            elif family == 'huggingface':
                return self._huggingface_response(client, messages, temperature, model_config)
            else:
                raise ValueError(f"Unsupported provider family: {family}")
            
        except Exception as e:
            print(f"Error getting response from {provider}: {str(e)}")
            return f"Failed to get response from {provider}: {str(e)}"

    def _openai_response(self, client: OpenAI, messages: List[Dict], temperature: float, model_config: Dict) -> str:
        """Get response from OpenAI"""
        try:
            response = client.chat.completions.create(
                model=model_config['name'],
                messages=messages,
                temperature=temperature
            )
            
            # Access API response fields with their original names
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._update_usage_stats(input_tokens, output_tokens, model_config)
            result = response.choices[0].message.content
            
            self._log_response("OpenAI", input_tokens, output_tokens, cost, result)
            return result
            
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return None

    def _anthropic_response(self, client: Anthropic, messages: List[Dict], temperature: float, model_config: Dict) -> str:
        """Get response from Anthropic"""
        try:
            # Convert messages to Anthropic format
            prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            response = client.messages.create(
                model=model_config['name'],
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.content[0].text
            
            # Get actual token counts from the response
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._update_usage_stats(input_tokens, output_tokens, model_config)
            
            self._log_response("Anthropic", input_tokens, output_tokens, cost, result)
            return result
            
        except Exception as e:
            print(f"Anthropic API error: {str(e)}")
            return None

    def _gemini_response(self, client: Any, messages: List[Dict], temperature: float, model_config: Dict) -> str:
        """Get response from Google's Gemini"""
        try:
            # Convert messages to Gemini format
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            response = client.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            
            result = response.text
            #print(f"Gemini response: {response}")
            
            # Get actual token counts from usage_metadata
            input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') and hasattr(response.usage_metadata, 'prompt_token_count') else 0
            output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') and hasattr(response.usage_metadata, 'candidates_token_count') else 0
            
            cost = self._update_usage_stats(input_tokens, output_tokens, model_config)
            
            self._log_response("Gemini", input_tokens, output_tokens, cost, result)
            return result
            
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return None

    def _mistral_response(self, client: Any, messages: List[Dict], temperature: float, model_config: Dict) -> str:
        """Get response from Mistral"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_keys['mistral']}"
            }
            
            data = {
                "model": model_config['name'],
                "messages": messages,
                "temperature": temperature
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data
            ).json()
            
            result = response['choices'][0]['message']['content']
            # Access API response fields with their original names
            input_tokens = response['usage']['prompt_tokens']
            output_tokens = response['usage']['completion_tokens']
            cost = self._update_usage_stats(input_tokens, output_tokens, model_config)
            
            self._log_response("Mistral", input_tokens, output_tokens, cost, result)
            return result
            
        except Exception as e:
            print(f"Mistral API error: {str(e)}")
            return None

    def _huggingface_response(self, client: InferenceClient, messages: List[Dict], temperature: float, model_config: Dict) -> str:
        """Get response from HuggingFace"""
        try:
            # Convert messages to a single prompt
            prompt = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            # Get input token count before generation
            input_tokens = client.count_input_tokens(prompt, model=model_config['name']).token_count
            
            # Generate response
            response = client.text_generation(
                prompt,
                model=model_config['name'],
                temperature=temperature,
                max_new_tokens=512,
                details=True  # Get detailed response including token counts
            )
            
            result = str(response.generated_text)
            
            # Get output token count
            output_tokens = response.details.generated_tokens
            
            cost = self._update_usage_stats(input_tokens, output_tokens, model_config)
            
            self._log_response("HuggingFace", input_tokens, output_tokens, cost, result)
            return result
            
        except Exception as e:
            print(f"HuggingFace API error: {str(e)}")
            return None

    def _deepseek_response(self, client: OpenAI, messages: List[Dict], temperature: float, model_config: Dict) -> str:
        """Get response from DeepSeek"""
        try:
            # Verify we're using the correct model
            model_name = model_config['name']
            if not model_name in ['deepseek-chat', 'deepseek-reasoner']:
                raise ValueError(f"Invalid DeepSeek model: {model_name}")
            
            print(f"\nUsing DeepSeek model: {model_name}")  # Debug log
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature
            )
            
            if not response:
                raise ValueError("Empty response received from DeepSeek API")
                
            print(f"Raw response: {response}")  # Debug log
            
            # Access API response fields with their original names
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._update_usage_stats(input_tokens, output_tokens, model_config)
            result = response.choices[0].message.content
            
            if not result:
                raise ValueError("Empty result content received from DeepSeek API")
            
            self._log_response("DeepSeek", input_tokens, output_tokens, cost, result)
            return result
            
        except Exception as e:
            print(f"DeepSeek API error: {str(e)}")
            print(f"Full error details: {repr(e)}")  # Print full error details
            return None

    def _groq_response(self, client: groq.Groq, messages: List[Dict], temperature: float, model_config: Dict) -> str:
        """Get response from Groq"""
        try:
            response = client.chat.completions.create(
                model=model_config['name'],
                messages=messages,
                temperature=temperature
            )
            
            # Access API response fields with their original names
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = input_tokens + output_tokens
            cost = self._update_usage_stats(input_tokens, output_tokens, model_config)
            result = response.choices[0].message.content
            
            self._log_response("Groq", input_tokens, output_tokens, cost, result)
            return result
            
        except Exception as e:
            print(f"Groq API error: {str(e)}")
            return None