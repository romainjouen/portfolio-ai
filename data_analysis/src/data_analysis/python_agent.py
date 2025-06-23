import time
from typing import Dict, Tuple, Optional, Any
import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from .python_agent_utils import BasePythonAgent, load_config


class PythonAgent(BasePythonAgent):
    """Python agent that works with any LLM provider based on model name."""
    
    def __init__(self, model_name: str, temperature: Optional[float] = None):
        self.provider = self._detect_provider_from_model(model_name)
        self.temperature = temperature
        super().__init__(model_name)
    
    def _detect_provider_from_model(self, model_name: str) -> str:
        """Auto-detect provider based on model name patterns."""
        openai_patterns = ['gpt-', 'o1-', 'text-', 'davinci', 'curie', 'babbage', 'ada']
        return 'openai' if any(pattern in model_name.lower() for pattern in openai_patterns) else 'ollama'
    
    def _get_default_model_name(self) -> str:
        """Get the default model name based on provider."""
        config = load_config()
        models_config = config.get('models', {})
        
        defaults = {'openai': 'gpt-4o-mini', 'ollama': 'qwen2.5-coder:32b'}
        return models_config.get(self.provider, {}).get('default', defaults[self.provider])
    
    @property
    def model(self):
        """Lazy load the appropriate model based on provider."""
        if self._model is None:
            try:
                # Use instance temperature if provided, otherwise fallback to config
                if self.temperature is not None:
                    temperature = self.temperature
                else:
                    config = load_config()
                    temperature = config.get('temperature', 0.0)
                
                if self.provider == 'openai':
                    from langchain_openai import ChatOpenAI
                    self._model = ChatOpenAI(model=self.model_name, temperature=temperature)
                else:  # ollama
                    from langchain_ollama.llms import OllamaLLM
                    self._model = OllamaLLM(model=self.model_name, temperature=temperature)
                    
            except ImportError:
                st.error(f"{self.provider.title()} dependencies not installed. Please install langchain-{self.provider}.")
            except Exception as e:
                st.error(f"Failed to load {self.provider} model {self.model_name}: {str(e)}")
        return self._model
    
    def generate_code(self, df: pd.DataFrame, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """Generate Python code using the configured model."""
        if not self.model:
            return "", {}
            
        start_time = time.time()
        df_info = self._analyze_dataframe(df)
        
        inputs = {
            "df_columns": df_info["columns"],
            "df_dtypes": df_info["dtypes"],
            "df_sample": df_info["sample_rows"],
            "df_shape": df_info["shape"],
            "query": user_query
        }
        
        # Generate code
        prompt = ChatPromptTemplate.from_template(self._create_python_prompt())
        result = (prompt | self.model).invoke(inputs)
        
        # Calculate metadata
        end_time = time.time()
        output_text = result.content if hasattr(result, 'content') else str(result)
        
        metadata = {
            "start_time": time.strftime("%H:%M:%S", time.localtime(start_time)),
            "end_time": time.strftime("%H:%M:%S", time.localtime(end_time)),
            "duration_seconds": round(end_time - start_time, 2),
            "input_tokens": len(prompt.format(**inputs).split()),
            "output_tokens": len(output_text.split()),
            "dataframe_info": df_info,
            "model_name": self.model_name,
            "provider": self.provider
        }
        
        return self._extract_code(result), metadata


 